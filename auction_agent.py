import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field
from omni_tools import omni_tools, OmniToolResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentType(str, Enum):
    LIST_PRODUCTS = "list_products"
    GET_PRODUCT_INFO = "get_product_info"
    PLACE_BID = "place_bid"
    GET_BID_HISTORY = "get_bid_history"
    HELP = "help"

class UserIntent(BaseModel):
    type: IntentType
    product_id: Optional[str] = None
    amount: Optional[float] = None
    user_id: Optional[str] = "user"
    session_id: Optional[str] = None
    confidence: float = 1.0

class ProductInfo(BaseModel):
    """Represents product information from the API"""
    id: str
    name: str
    description: str
    current_highest_bid: float
    auction_end_time: str
    time_remaining: str
    
    def format_for_voice(self) -> str:
        """Format product info for voice response"""
        return (
            f"{self.name}: {self.description}\n"
            f"Current highest bid: ${self.current_highest_bid:.2f}\n"
            f"{self.time_remaining}"
        )

class BidInfo(BaseModel):
    """Represents bid information"""
    user: str
    amount: float
    timestamp: str
    status: str

class AuctionAgent:
    """Handles auction operations and voice interactions using OmniDimension tools."""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.last_products_fetch: Optional[datetime] = None
        self.products_cache: List[Dict[str, Any]] = []
        self.products: List[Dict[str, Any]] = []  # Store product data
        self.cache_ttl = 30  # seconds
    
    async def list_products(self, session_id: Optional[str] = None) -> str:
        """Fetch and list all available auction products"""
        try:
            # Check cache first
            current_time = datetime.now()
            if (self.last_products_fetch is None or 
                (current_time - self.last_products_fetch).total_seconds() > self.cache_ttl):
                result = await omni_tools.list_products()
                if result.status == "success":
                    self.products_cache = result.data.get("products", [])
                    self.products = self.products_cache  # Update products list
                    self.last_products_fetch = current_time
                else:
                    return "I'm having trouble fetching the products. Please try again later."
            
            if not self.products_cache:
                return "There are currently no products available for auction."
                
            return await omni_tools.format_product_list(self.products_cache)
            
        except Exception as e:
            logger.error(f"Error listing products: {str(e)}")
            return "I'm sorry, I couldn't fetch the product list. Please try again later."
    
    async def get_product_info(self, product_id: str) -> str:
        """Get detailed information about a specific product"""
        try:
            if not product_id:
                return "Please specify a product ID."
                
            result = await omni_tools.get_product(product_id)
            if result.status != "success" or not result.data:
                return f"I couldn't find product with ID {product_id}. Please try again with a different ID."
            
            product = ProductInfo(**result.data)
            return product.format_for_voice()
            
        except Exception as e:
            logger.error(f"Error getting product info: {str(e)}")
            return "I'm sorry, I couldn't fetch the product information. Please try again later."
    
    async def place_bid(self, product_id: str, amount: float, user_id: str = "user") -> str:
        """Place a bid on a product"""
        try:
            if not product_id:
                return "Please specify a product ID."
                
            if not amount or amount <= 0:
                return "Please specify a valid bid amount greater than zero."
                
            # Get current product info to validate
            product_result = await omni_tools.get_product(product_id)
            if product_result.status != "success":
                return "I couldn't find that product. Please check the product ID and try again."
                
            current_bid = product_result.data.get("current_highest_bid", 0)
            if amount <= current_bid:
                return f"Your bid must be higher than the current highest bid of ${current_bid:.2f}."
                
            # Place the bid
            bid_result = await omni_tools.place_bid(product_id, user_id, amount)
            if bid_result.status != "success":
                return "I couldn't place your bid. Please try again."
                
            return f"Your bid of ${amount:.2f} has been placed successfully!"
            
        except Exception as e:
            logger.error(f"Error placing bid: {str(e)}")
            return "I'm sorry, I couldn't process your bid. Please try again later."
    
    async def get_bid_history(self, product_id: str) -> str:
        """Get bid history for a product"""
        try:
            if not product_id:
                return "Please specify a product ID."
                
            result = await omni_tools.get_bid_history(product_id)
            if result.status != "success":
                return "I couldn't fetch the bid history. Please try again later."
                
            bids = result.data.get("bids", [])
            if not bids:
                return "No bid history found for this product."
                
            response = "Here's the bid history for this product:\n"
            for bid in sorted(bids, key=lambda x: x.get("timestamp", ""), reverse=True)[:5]:
                bid_info = BidInfo(**bid)
                response += f"- ${bid_info.amount:.2f} by {bid_info.user} at {bid_info.timestamp}\n"
                
            return response
            
        except Exception as e:
            logger.error(f"Error getting bid history: {str(e)}")
            return "I'm sorry, I couldn't fetch the bid history. Please try again later."

    async def process_voice_command(self, intent: UserIntent) -> str:
        """Process voice command based on detected intent"""
        try:
            if intent.type == IntentType.LIST_PRODUCTS:
                return await self.list_products()
                
            elif intent.type == IntentType.GET_PRODUCT_INFO:
                if not intent.product_id:
                    return "Please specify which product you'd like information about."
                return await self.get_product_info(intent.product_id)
                
            elif intent.type == IntentType.PLACE_BID:
                if not intent.product_id:
                    return "Please specify which product you'd like to bid on."
                if not intent.amount or intent.amount <= 0:
                    return "Please specify a valid bid amount greater than zero."
                return await self.place_bid(intent.product_id, intent.amount, intent.user_id)
                
            elif intent.type == IntentType.GET_BID_HISTORY:
                if not intent.product_id:
                    return "Please specify which product's bid history you'd like to see."
                return await self.get_bid_history(intent.product_id)
                
            elif intent.type == IntentType.HELP:
                return self._get_help_response()
                
            return "I'm not sure how to help with that. Please try rephrasing your request."
            
        except Exception as e:
            logger.error(f"Error processing voice command: {str(e)}")
            return "I'm sorry, I encountered an error processing your request. Please try again."
    
    def _get_help_response(self) -> str:
        """Get help message with available commands"""
        return (
            "I can help you with the following:\n"
            "- List all available auction items\n"
            "- Get information about a specific product\n"
            "- Place a bid on an item\n"
            "- Check bid history for a product\n"
            "Just let me know what you'd like to do!"
        )

# Singleton instance
auction_agent = AuctionAgent()

# Example usage for testing
async def test():
    # List products
    print(await auction_agent.list_products())
    
    # Get product info (replace '1' with actual product ID)
    print(await auction_agent.get_product_info("1"))
    
    # Place a bid (replace '1' with actual product ID and 100.0 with bid amount)
    print(await auction_agent.place_bid("1", 100.0, "test_user"))
    
    # Get bid history (replace '1' with actual product ID)
    print(await auction_agent.get_bid_history("1"))

if __name__ == "_main_":
    import asyncio
    asyncio.run(test())