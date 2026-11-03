import pandas as pd
import numpy as np
import random
import os

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Sample text components for realistic support tickets
ticket_templates = {
    'Technical': [
        "App keeps crashing whenever I try to click on the checkout button.",
        "Unable to login to my account, getting unknown server error 500.",
        "The latest update broke the dashboard loading time. Extremely slow.",
        "Payment gateway timeout error while completing the transaction.",
        "Feature X is completely non-responsive on mobile devices."
    ],
    'Billing': [
        "I was double charged for my monthly subscription order.",
        "Requesting a full refund as the service was unavailable all weekend.",
        "Invoice details are incorrect, tax calculation shows wrong amount.",
        "Payment deducted twice from bank account but subscription not active.",
        "Cannot update my credit card details in the billing settings page."
    ],
    'Product Inquiry': [
        "Is there a discount available for annual enterprise subscriptions?",
        "Does the system support integration with external REST APIs?",
        "How can I export my weekly analytics reports into CSV or PDF format?",
        "Are there any limits on the number of team members under the basic plan?",
        "Where can I find the official API documentation for webhook triggers?"
    ],
    'Account Management': [
        "Need help resetting my two-factor authentication device settings.",
        "How do I transfer ownership of my workspace to another admin?",
        "Requesting permanent deletion of my account and stored personal data.",
        "User permissions are not updating after changing role in admin panel.",
        "Unable to verify my email address despite clicking the confirmation link."
    ]
}

priorities = ['Low', 'Medium', 'High', 'Critical']
channels = ['Email', 'Web Portal', 'Mobile App', 'Chatbot']

data = []
for i in range(1, 5001):
    category = random.choice(list(ticket_templates.keys()))
    base_text = random.choice(ticket_templates[category])
    
    # Add minor variations
    ticket_id = f"TICK-{i:05d}"
    priority = random.choice(priorities)
    channel = random.choice(channels)
    
    data.append({
        'Ticket_ID': ticket_id,
        'Category': category,
        'Ticket_Text': base_text,
        'Priority': priority,
        'Channel': channel
    })

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to Data/raw folder
output_path = os.path.join('Data', 'raw', 'customer_tickets_raw.csv')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Successfully generated 5,000 support tickets and saved to {output_path}!")