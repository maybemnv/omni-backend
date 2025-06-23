from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import requests
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BidStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    INVALID = "invalid"

class OmniToolResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

class OmniTools:
    def _init_(self, base_url: str = "https://web-production-0a66.up.railway.app"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    async def list_products(self) -> OmniToolResponse:
        """Fetch all available auction products"""
        try:
            response = self.session.get(f"{self.base_url}/api/products")
            response.raise_for_status()
            return OmniToolResponse(
                status="success",
                message="Products retrieved successfully",
                data={"products": response.json()}
            )
        except Exception as e:
            logger.error(f"Error listing products: {str(e)}")
            return OmniToolResponse(
                status="error",
                message=f"Failed to fetch products: {str(e)}"
            )
    
    async def get_product(self, product_id: str) -> OmniToolResponse:
        """Get details of a specific product"""
        try:
            response = self.session.get(f"{self.base_url}/api/products/{product_id}")
            response.raise_for_status()
            return OmniToolResponse(
                status="success",
                message="Product details retrieved",
                data=response.json()
            )
        except Exception as e:
            logger.error(f"Error getting product {product_id}: {str(e)}")
            return OmniToolResponse(
                status="error",
                message=f"Failed to get product details: {str(e)}"
            )
    
    async def place_bid(
        self, 
        product_id: str, 
        user_id: str, 
        amount: float
    ) -> OmniToolResponse:
        """Place a bid on a product"""
        try:
            payload = {
                "user": user_id,
                "amount": amount,
                "product_id": product_id
            }
            response = self.session.post(
                f"{self.base_url}/api/bids",
                json=payload
            )
            response.raise_for_status()
            
            return OmniToolResponse(
                status="success",
                message="Bid placed successfully",
                data=response.json()
            )
        except Exception as e:
            logger.error(f"Error placing bid: {str(e)}")
            return OmniToolResponse(
                status="error",
                message=f"Failed to place bid: {str(e)}"
            )
    
    async def get_bid_history(self, product_id: str) -> OmniToolResponse:
        """Get bid history for a product"""
        try:
            response = self.session.get(f"{self.base_url}/api/products/{product_id}/bids")
            response.raise_for_status()
            return OmniToolResponse(
                status="success",
                message="Bid history retrieved",
                data={"bids": response.json()}
            )
        except Exception as e:
            logger.error(f"Error getting bid history: {str(e)}")
            return OmniToolResponse(
                status="error",
                message=f"Failed to get bid history: {str(e)}"
            )

    async def format_product_list(self, products: List[Dict]) -> str:
        """Format product list for voice response"""
        if not products:
            return "No products available at the moment."
            
        response = "Here are the available products:\n"
        for product in products:
            time_remaining = product.get('time_remaining', 'N/A')
            response += (
                f"- {product.get('name', 'Unknown')}: "
                f"${product.get('current_highest_bid', 0):.2f} "
                f"({time_remaining})\n"
            )
        return response

# Singleton instance
omni_tools = OmniTools()