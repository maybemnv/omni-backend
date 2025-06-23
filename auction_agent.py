import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional

@dataclass
class Bid:
    user: str
    amount: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Product:
    id: str
    name: str
    description: str
    current_highest_bid: float = 0.0
    auction_end_time: datetime = field(default_factory=lambda: datetime.now() + timedelta(minutes=10))
    bidding_history: List[Bid] = field(default_factory=list)
    
    def time_remaining(self) -> str:
        remaining = self.auction_end_time - datetime.now()
        if remaining.total_seconds() <= 0:
            return "Auction has ended"
        minutes, seconds = divmod(int(remaining.total_seconds()), 60)
        return f"{minutes}m {seconds}s remaining"
    
    def place_bid(self, user: str, amount: float) -> str:
        if datetime.now() > self.auction_end_time:
            return "Error: Auction has already ended"
            
        if amount <= self.current_highest_bid:
            return f"Error: Bid must be higher than current highest bid (${self.current_highest_bid})"
            
        self.current_highest_bid = amount
        bid = Bid(user=user, amount=amount)
        self.bidding_history.append(bid)
        return f"Success! Your bid of ${amount:.2f} on {self.name} has been placed."

class AuctionAgent:
    def __init__(self):
        self.products: Dict[str, Product] = {
            "iphone": Product(
                id="1",
                name="iPhone 15 Pro",
                description="Latest iPhone with A17 Pro chip and 48MP camera",
                current_highest_bid=1000.0,
                auction_end_time=datetime.now() + timedelta(minutes=15)
            ),
            "macbook": Product(
                id="2",
                name="MacBook Pro 16",
                description="M2 Max chip with 12-core CPU and 38-core GPU",
                current_highest_bid=2000.0,
                auction_end_time=datetime.now() + timedelta(minutes=30)
            ),
            "airpods": Product(
                id="3",
                name="AirPods Pro",
                description="Wireless earbuds with active noise cancellation",
                current_highest_bid=500.0,
                auction_end_time=datetime.now() + timedelta(minutes=20)
            ),
            "google_pixel_7": Product(
                id="4",
                name="Google Pixel 7",
                description="Latest Google Pixel with 50MP camera and 120Hz display",
                current_highest_bid=1500.0,
                auction_end_time=datetime.now() + timedelta(minutes=10)
            ),
            "RTX 5090": Product(
                id="5",
                name="RTX 5090",
                description="Latest NVIDIA RTX 5090 with 24GB GDDR6X memory",
                current_highest_bid=3000.0,
                auction_end_time=datetime.now() + timedelta(minutes=10)
            )
        }
    
    def list_products(self) -> str:
        response = "Current Auction Items:\n"
        for product in self.products.values():
            response += f"- {product.name}: ${product.current_highest_bid:.2f} ({product.time_remaining()})\n"
        return response
    
    def get_product_info(self, product_name: str) -> str:
        product = self._find_product(product_name)
        if not product:
            return "Product not found. Please check the product name."
            
        return (
            f"{product.name}: {product.description}\n"
            f"Current Bid: ${product.current_highest_bid:.2f}\n"
            f"{product.time_remaining()}"
        )
    
    def place_bid(self, product_name: str, amount: float, user: str = "User") -> str:
        product = self._find_product(product_name)
        if not product:
            return "Product not found. Please check the product name."
            
        return product.place_bid(user, amount)
    
    def _find_product(self, product_name: str) -> Optional[Product]:
        product_name = product_name.lower()
        for key, product in self.products.items():
            if product_name in key or product_name in product.name.lower():
                return product
        return None

def main():
    agent = AuctionAgent()
    print("Welcome to OmniAuction!")
    print("You can interact with the auction using voice commands.")
    print("Type 'help' to see available commands.\n")
    
    while True:
        try:
            command = input("\nWhat would you like to do? ").strip().lower()
            
            if command in ['exit', 'quit']:
                print("Thank you for using OmniAuction!")
                break
                
            elif command == 'help':
                print("\nAvailable commands:")
                print("- 'list': List all auction items")
                print("- 'info [product]': Get info about a product")
                print("- 'bid [amount] on [product]': Place a bid")
                print("- 'exit' or 'quit': Exit the program")
                
            elif command == 'list':
                print(agent.list_products())
                
            elif command.startswith('info '):
                product_name = command[5:].strip()
                print(agent.get_product_info(product_name))
                
            elif 'bid' in command and 'on' in command:
                try:
                    parts = command.split()
                    amount = float(parts[1])
                    product_name = ' '.join(parts[3:])
                    print(agent.place_bid(product_name, amount))
                except (IndexError, ValueError):
                    print("Invalid bid format. Use: 'bid [amount] on [product]'")
                    
            else:
                print("I didn't understand that command. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\nThank you for using OmniAuction!")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
