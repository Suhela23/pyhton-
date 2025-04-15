import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from dotenv import load_dotenv
import os
import openai
import json
import google.generativeai as genai
import re

# Load environment variables
load_dotenv()

# Set up API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if API keys are available and valid
use_gemini = False
use_openai = False

if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)
        use_gemini = True
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        use_gemini = False

if openai_api_key:
    try:
        openai.api_key = openai_api_key
        use_openai = True
    except Exception as e:
        print(f"Error configuring OpenAI API: {e}")
        use_openai = False

# Set page title and configuration
st.set_page_config(
    page_title="Enhanced Customer Support Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Define custom CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #2b313e
    }
    .chat-message.bot {
        background-color: #475063
    }
    .chat-message .avatar {
      width: 20%;
    }
    .chat-message .avatar img {
      max-width: 78px;
      max-height: 78px;
      border-radius: 50%;
      object-fit: cover;
    }
    .chat-message .message {
      width: 80%;
      padding: 0 1.5rem;
    }
    a {
      color: #8ab4f8 !important;
      text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# Load BERT model and tokenizer
@st.cache_resource
def load_bert_model():
    try:
        # Load pre-trained BERT model for sequence classification
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=7)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading BERT model: {e}")
        return None, None

# Load product and company information for context
@st.cache_data
def load_company_data():
    # This could come from a database in a real applicationpython3 "c:/Users/Suhela Thasleem/Downloads/app.py"

    company_data = {
        "company_name": "TechShop",
        "products": [
            {"name": "SmartPhone X", "price": "$899", "features": "5G, 128GB storage, 6.5 inch screen"},
            {"name": "Laptop Pro", "price": "$1299", "features": "16GB RAM, 512GB SSD, 14 inch display"},
            {"name": "Wireless Earbuds", "price": "$129", "features": "Noise cancellation, 24hr battery life"},
            {"name": "Smart Watch", "price": "$249", "features": "Heart rate monitor, GPS, 5-day battery"}
        ],
        "policies": {
            "returns": "30-day free returns on all products. Must be in original packaging.",
            "shipping": "Free shipping on orders over $50. Standard shipping takes 3-5 business days.",
            "warranty": "All products come with a 1-year limited warranty."
        },
        "contact": {
            "email": "support@techshop.example",
            "phone": "1-800-TECH-HELP",
            "hours": "Monday-Friday 8am-8pm EST"
        },
        "competitors": {
            "Amazon": "https://www.amazon.com",
            "Best Buy": "https://www.bestbuy.com",
            "Walmart": "https://www.walmart.com",
            "NewEgg": "https://www.newegg.com",
            "B&H Photo Video": "https://www.bhphotovideo.com"
        }
    }
    return company_data

# Add a template-based response system that doesn't require APIs
def get_template_response(query, company_data):
    """
    Generate responses based on templates when APIs are unavailable
    """
    query_lower = query.lower()
    
    # Check for product inquiries first
    product_name = extract_product_name(query)
    if product_name:
        # Check if this is a product we have
        our_products = [p["name"].lower() for p in company_data["products"]]
        if any(product.lower() in product_name.lower() for product in our_products):
            # Find the matching product
            for product in company_data["products"]:
                if product["name"].lower() in product_name.lower():
                    return f"""Yes, we carry the {product["name"]}! 
                    
                    It's priced at {product["price"]} and features {product["features"]}.
                    
                    Would you like to know more about this product or are you interested in purchasing it?"""
        else:
            # Product not in our inventory
            competitor_links = []
            for name, url in company_data["competitors"].items():
                competitor_links.append(f'<a href="{url}" target="_blank">{name}</a>')
            
            competitor_text = ", ".join(competitor_links[:-1]) + " or " + competitor_links[-1] if len(competitor_links) > 1 else competitor_links[0]
            
            return f"""I'm sorry, but we don't currently carry {product_name} in our inventory at {company_data['company_name']}. 
            
            You might be able to find it at one of these retailers: {competitor_text}.
            
            If you're looking for similar products that we do carry, I'd be happy to recommend some alternatives from our current inventory."""
    
    # Check for return policy questions
    if any(word in query_lower for word in ["return", "send back", "refund", "exchange"]):
        return f"Our return policy: {company_data['policies']['returns']}"
    
    # Check for shipping questions
    if any(word in query_lower for word in ["shipping", "delivery", "ship", "arrive"]):
        return f"Our shipping policy: {company_data['policies']['shipping']}"
    
    # Check for warranty questions
    if any(word in query_lower for word in ["warranty", "guarantee", "repair"]):
        return f"Our warranty policy: {company_data['policies']['warranty']}"
    
    # Check for contact information
    if any(word in query_lower for word in ["contact", "phone", "email", "reach", "hours"]):
        return f"""You can contact our customer support team at:
        
        📧 Email: {company_data['contact']['email']}
        ☎️ Phone: {company_data['contact']['phone']}
        ⏰ Hours: {company_data['contact']['hours']}"""
    
    # Check for general product catalog
    if any(word in query_lower for word in ["catalog", "products", "sell", "offer", "inventory"]):
        product_list = "\n".join([f"• {p['name']} - {p['price']} - {p['features']}" for p in company_data['products']])
        return f"""Here are the products we currently offer:
        
        {product_list}
        
        Is there a specific product you're interested in?"""
    
    # Default response
    return """Thank you for contacting our customer support. I can help with information about our products, return policies, shipping details, and more.
    
    How can I assist you today?"""

# Function to get response from OpenAI API
def get_openai_response(query, company_data):
    try:
        # Create a context with company data
        context = json.dumps(company_data)
        
        # Use OpenAI to generate a response
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"""You are a helpful customer support chatbot for {company_data['company_name']}. 
                    Use the following company information to provide accurate answers: {context}.
                    
                    If a customer asks about a product that is not in our inventory, kindly let them know we don't carry it, 
                    but suggest one of our competitor websites from the competitors list where they might find it, with a clickable link.
                    Format the link as proper HTML so it's clickable in the chat interface."""},
                    {"role": "user", "content": query}
                ],
                max_tokens=350,
                temperature=0.7
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            # Fallback to older API if chat completions don't work
            response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",
                prompt=f"""You are a helpful customer support chatbot for {company_data['company_name']}. 
                Use the following company information to provide accurate answers: {context}.
                
                If a customer asks about a product that is not in our inventory, kindly let them know we don't carry it, 
                but suggest one of our competitor websites from the competitors list where they might find it, with a clickable link.
                Format the link as proper HTML so it's clickable in the chat interface.
                
                Customer: {query}
                
                Chatbot:""",
                max_tokens=350,
                temperature=0.7
            )
            return response.choices[0].text.strip()
            
    except Exception as e:
        st.error(f"Error with OpenAI API: {e}")
        return f"I'm having trouble connecting to my knowledge base. Please try again or contact our support team at {company_data['contact']['email']}."

# Function to get integrated response from available AI model
def get_llm_response(query, company_data):
    # First try to use APIs if available
    if use_gemini:
        try:
            return get_gemini_response(query, company_data)
        except Exception as e:
            print(f"Error with Gemini API: {e}")
            # Fall through to next option
    
    if use_openai:
        try:
            return get_openai_response(query, company_data)
        except Exception as e:
            print(f"Error with OpenAI API: {e}")
            # Fall through to fallback
    
    # If we got here, both APIs failed or weren't configured
    # Use our built-in template-based response system
    return get_template_response(query, company_data)

# Function to process the response and make any links clickable
def process_response(response):
    # Make URLs clickable if they aren't already in HTML format
    if '<a href=' not in response:
        # Find URLs that aren't already in HTML tags
        url_pattern = r'(?<![\">])(https?://[^\s<>\'"]+)'
        response = re.sub(url_pattern, r'<a href="\1" target="_blank">\1</a>', response)
    
    return response

# Function to extract product name from query
def extract_product_name(query):
    # Simple keyword extraction - in a real system, use NER or more advanced techniques
    query_lower = query.lower()
    
    # List of keywords that might indicate a product inquiry
    product_indicators = ["where can i find", "do you sell", "do you have", "looking for", 
                         "searching for", "interested in", "want to buy", "purchase"]
    
    # Check if query contains any product inquiry indicators
    contains_product_inquiry = any(indicator in query_lower for indicator in product_indicators)
    
    if contains_product_inquiry:
        # Use BERT or other NLP techniques to extract the product name
        # For simplicity, we'll use basic substring extraction
        for indicator in product_indicators:
            if indicator in query_lower:
                # Get text after the indicator
                product_text = query_lower.split(indicator, 1)[1].strip()
                # Remove common articles and clean up
                product_text = product_text.replace("a ", "").replace("an ", "").replace("the ", "")
                # Remove punctuation at the end
                product_text = product_text.rstrip("?,.!")
                return product_text
    
    return None

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main function
def main():
    st.title("Enhanced Customer Support Chatbot")
    
    # Load model and data
    tokenizer, model = load_bert_model()
    company_data = load_company_data()
    
    # Add a sidebar with information
    with st.sidebar:
        st.subheader(f"About {company_data['company_name']}")
        st.write("This is an AI-powered customer support chatbot that can help with:")
        st.write("✅ Product information")
        st.write("✅ Order status inquiries")
        st.write("✅ Return policies")
        st.write("✅ Technical support")
        st.write("✅ General questions")
        st.write("✅ Alternative product sources")
        
        # Show which APIs are available
        if use_openai or use_gemini:
            api_status = []
            if use_openai:
                api_status.append("BERT: ✅")
            else:
                api_status.append("BERT: ❌")
                
            if use_gemini:
                api_status.append("BERT: ✅")
            else:
                api_status.append("BERT: ❌")
                
            st.info("\n".join(api_status))
            
            if not use_openai and not use_gemini:
                st.warning("No API keys configured. Using built-in template responses.")
        else:
            st.warning("No API keys configured. Using built-in template responses.")
        
        if st.checkbox("Show company data", False):
            st.json(company_data)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.container():
            st.markdown(f"""
            <div class="chat-message {message['role']}">
                <div class="avatar">
                    <img src="{'https://api.dicebear.com/7.x/bottts/svg?seed=Lily' if message['role'] == 'bot' else 'https://api.dicebear.com/7.x/personas/svg?seed=Felix'}">
                </div>
                <div class="message">{message['content']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # User input
    user_query = st.chat_input("Ask me anything about our products, policies, or support...")
    
    if user_query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.container():
            st.markdown(f"""
            <div class="chat-message user">
                <div class="avatar">
                    <img src="https://api.dicebear.com/7.x/personas/svg?seed=Felix">
                </div>
                <div class="message">{user_query}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Get response from bot
        with st.spinner("Thinking..."):
            # Check if query might be about a product we don't have
            product_name = extract_product_name(user_query)
            
            # Generate dynamic response using LLM
            raw_response = get_llm_response(user_query, company_data)
            
            # Process response to ensure links are clickable
            processed_response = process_response(raw_response)
            
            # Add bot response to chat history
            st.session_state.messages.append({"role": "bot", "content": processed_response})
            
            # Display bot response
            with st.container():
                st.markdown(f"""
                <div class="chat-message bot">
                    <div class="avatar">
                        <img src="https://api.dicebear.com/7.x/bottts/svg?seed=Lily">
                    </div>
                    <div class="message">{processed_response}</div>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()