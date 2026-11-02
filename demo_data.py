"""
demo_data.py – Six demo tickets for ResolveIQ.
Tickets 1-3 share a payment-outage theme and MUST form one suspected incident.
"""
from datetime import datetime, timedelta

# Base time so all tickets fall within the 24-hour clustering window
BASE_TIME = datetime.now() - timedelta(minutes=30)

DEMO_TICKETS = [
    {
        "customer_name": "Alice Thompson",
        "channel": "Email",
        "message": "Our payment gateway is failing. Customers cannot complete transactions.",
        "offset_min": 0,
    },
    {
        "customer_name": "Bob Martinez",
        "channel": "Chat",
        "message": "My card transaction is showing an error during checkout.",
        "offset_min": 5,
    },
    {
        "customer_name": "Carol Singh",
        "channel": "Email",
        "message": "The payment page is not responding.",
        "offset_min": 10,
    },
    {
        "customer_name": "David Okafor",
        "channel": "Email",
        "message": "I was charged twice for my monthly subscription.",
        "offset_min": 15,
    },
    {
        "customer_name": "Eva Chen",
        "channel": "Chat",
        "message": "How can I reset my account password?",
        "offset_min": 20,
    },
    {
        "customer_name": "Frank Russo",
        "channel": "Email",
        "message": "The application crashes after the latest update.",
        "offset_min": 25,
    },
]
