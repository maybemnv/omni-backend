import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta
import json
import os
import uvicorn
import redis.asyncio as redis
from starlette.requests import Request
from starlette.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from auction_agent import AuctionAgent, Product, Bid

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="OmniAuction API",
    description="REST API for OmniAuction Voice Agent",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add rate limiter to the app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize Redis for rate limiting
@app.on_event("startup")
async def startup():
    redis_connection = redis.from_url("redis://localhost:6379", encoding="utf-8", decode_responses=True)
    await FastAPILimiter.init(redis_connection)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:8000",  # Local FastAPI
        "https://your-production-domain.com"  # Replace with your production domain
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"]
)

# WebSocket manager
class ConnectionManager:
    def _init_(self):
        self.active_connections: List[WebSocket] = []
        self.agent = AuctionAgent()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict) -> None:
        """Broadcast message to all active WebSocket connections.
        
        Args:
            message: Dictionary containing the message to broadcast
        """
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except (WebSocketDisconnect, RuntimeError) as e:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            if connection in self.active_connections:
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Models
class BidRequest(BaseModel):
    """Request model for placing a bid."""
    user: str = Field(..., min_length=3, max_length=50, description="Unique user identifier")
    amount: float = Field(..., gt=0, description="Bid amount, must be greater than 0")
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Bid amount must be greater than 0')
        return round(v, 2)  # Ensure 2 decimal places for currency

class AutoBidRequest(BaseModel):
    """Request model for setting up auto-bidding."""
    user: str = Field(..., min_length=3, max_length=50, description="Unique user identifier")
    max_bid: float = Field(..., gt=0, description="Maximum bid amount")
    product_id: str = Field(..., min_length=1, description="ID of the product to bid on")

class VoiceCommand(BaseModel):
    """Model for voice command data."""
    text: str = Field(..., min_length=1, description="The voice command text")
    session_id: str = Field(..., min_length=1, description="Unique session identifier")

# API Endpoints
@app.get(
    "/api/products",
    response_model=List[Dict],
    summary="List all products",
    description="Retrieve a list of all available auction products",
    dependencies=[Depends(RateLimiter(times=100, hours=1))]
)
async def list_products() -> List[Dict[str, Any]]:
    """Get list of all auction products"""
    products = []
    for product_id, product in manager.agent.products.items():
        products.append({
            "id": product_id,
            "name": product.name,
            "description": product.description,
            "current_highest_bid": product.current_highest_bid,
            "time_remaining": product.time_remaining(),
            "bids_count": len(product.bidding_history),
            "auction_end_time": product.auction_end_time.isoformat()
        })
    return products

@app.get(
    "/api/products/{product_id}",
    response_model=Dict,
    summary="Get product details",
    description="Retrieve detailed information about a specific product",
    responses={
        404: {"description": "Product not found"},
        200: {"description": "Product details retrieved successfully"}
    }
)
async def get_product(product_id: str) -> Dict[str, Any]:
    """Get details of a specific product"""
    product = manager.agent._find_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    return {
        "id": product.id,
        "name": product.name,
        "description": product.description,
        "current_highest_bid": product.current_highest_bid,
        "time_remaining": product.time_remaining(),
        "bids_count": len(product.bidding_history),
        "bidding_history": [
            {"user": bid.user, "amount": bid.amount, "timestamp": bid.timestamp.isoformat()}
            for bid in product.bidding_history[-10:]
        ]
    }

@app.post(
    "/api/bids",
    status_code=status.HTTP_201_CREATED,
    summary="Place a new bid",
    description="Submit a new bid for a product",
    dependencies=[Depends(RateLimiter(times=10, minutes=1))],
    responses={
        201: {"description": "Bid placed successfully"},
        400: {"description": "Invalid bid amount or auction ended"},
        404: {"description": "Product not found"}
    }
)
async def place_bid(bid: BidRequest) -> Dict[str, str]:
    """Place a new bid on a product.
    
    Args:
        bid: BidRequest object containing user, amount, and product_id
        
    Returns:
        Dict with status and message
        
    Raises:
        HTTPException: If product not found, invalid bid, or server error
    """
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back the received message
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/products/{product_id}/bids/count", response_model=Dict[str, int])
async def get_bid_count(product_id: str):
    """Get total number of bids for a product"""
    product = manager.agent.get_product_by_id(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return {"count": len(product.bidding_history)}

@app.get("/api/products/{product_id}/bids", response_model=List[Dict])
async def get_bid_history(product_id: str):
    """Get full bid history for a product"""
    product = manager.agent.get_product_by_id(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    return [{
        "user": bid.user,
        "amount": bid.amount,
        "timestamp": bid.timestamp.isoformat()
    } for bid in product.bidding_history]

@app.post(
    "/api/products/{product_id}/auto-bid",
    response_model=Dict,
    summary="Set up auto-bidding",
    description="Configure auto-bidding for a user on a specific product",
    dependencies=[Depends(RateLimiter(times=5, minutes=10))],
    responses={
        200: {"description": "Auto-bid configured successfully"},
        400: {"description": "Invalid auto-bid configuration"},
        404: {"description": "Product not found"},
        429: {"description": "Too many requests"}
    }
)
async def set_auto_bid(
    product_id: str,
    auto_bid: AutoBidRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Set up auto-bidding for a user on a product.
    
    Args:
        product_id: ID of the product to set auto-bid for
        auto_bid: AutoBidRequest containing user and max bid amount
        
    Returns:
        Dict with status and auto-bid details
        
    Raises:
        HTTPException: If product not found or invalid auto-bid configuration
    """
    try:
        product = manager.agent.get_product_by_id(product_id)
        if not product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Product not found"
            )
        
        # Validate max bid amount
        if auto_bid.max_bid <= product.current_highest_bid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Auto-bid amount must be higher than current highest bid (${product.current_highest_bid:.2f})"
            )
        
        # Store auto-bid in the auction agent
        manager.agent.set_auto_bid(product_id, auto_bid.user, auto_bid.max_bid)
        
        # Start background task to check for auto-bid opportunities
        background_tasks.add_task(
            manager.agent.monitor_auto_bids,
            product_id=product_id
        )
        
        return {
            "status": "success",
            "message": (
                f"Auto-bid configured for {auto_bid.user} on {product.name} "
                f"up to ${auto_bid.max_bid:.2f}"
            ),
            "data": {
                "product_id": product_id,
                "user": auto_bid.user,
                "max_bid": auto_bid.max_bid,
                "product_name": product.name,
                "current_highest_bid": product.current_highest_bid,
                "time_remaining": product.time_remaining()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set up auto-bid: {str(e)}"
        )

# Background task to check for ending auctions
async def check_ending_auctions():
    while True:
        await asyncio.sleep(10)  # Check every 10 seconds
        for product in manager.agent.products.values():
            time_remaining = (product.auction_end_time - datetime.now()).total_seconds()
            if 0 < time_remaining < 60:  # Less than 1 minute remaining
                await manager.broadcast({
                    "type": "auction_ending_soon",
                    "product_id": product.id,
                    "product_name": product.name,
                    "time_remaining": f"{int(time_remaining)} seconds"
                })

@app.on_event("startup")
async def startup_event():
    # Add some sample products if none exist
    if not manager.agent.products:
        sample_products = [
            Product(
                id="prod_123",
                name="Vintage Camera",
                description="Classic film camera from the 1970s",
                starting_price=100.0,
                auction_end_time=datetime.utcnow() + timedelta(days=7)
            ),
            Product(
                id="prod_456",
                name="Smart Watch",
                description="Latest model smart watch with health tracking",
                starting_price=200.0,
                auction_end_time=datetime.utcnow() + timedelta(days=3)
            )
        ]
        manager.agent.products = sample_products
        print("Initialized sample products")
    
    # Start background task for checking ending auctions
    asyncio.create_task(check_ending_auctions())

def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application instance
    """
    return app

if _name_ == "_main_":
    import uvicorn
    
    # Configure logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('auction.log')
        ]
    )
    
    # Start the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV") == "development",
        log_level="info",
        proxy_headers=True,
        forwarded_allow_ips='*'
    )