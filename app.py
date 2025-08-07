import streamlit as st
import os
import sys
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --- Project Structure Setup ---
# This ensures that the app can find the other modules (config, models, utils)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# --- Module Imports ---
from models.llm import get_chatgroq_model
from models.embeddings import get_huggingface_embeddings
from utils.rag_utils import get_or_create_vector_store, get_context_from_rag
from utils.search_utils import perform_web_search

# --- NEW FEATURE: Function to format chat history for the LLM ---
def format_chat_history(messages):
    """Formats the chat history into a readable string for the LLM."""
    history = []
    for msg in messages:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history.append(f"{role}: {msg.content}")
    return "\n".join(history)

# --- Main App Logic ---

def chat_page():
    """Main chat interface page for the Rental Assistant"""
    st.title("🏡 AI Rental Search Assistant")
    st.markdown("I can help you find rental properties. Ask me about neighborhoods, listings, and more!")

    # --- Sidebar for Settings ---
    with st.sidebar:
        st.header("Settings")
        response_mode = st.radio(
            "Response Mode",
            ("Detailed", "Concise"),
            help="Choose 'Detailed' for full explanations or 'Concise' for quick summaries."
        )

        st.divider()

        # --- NEW FEATURE: Neighborhood Match Score ---
        st.header("Neighborhood Matcher")
        st.session_state.user_preferences = st.text_area(
            "Tell me what you're looking for in a neighborhood:",
            placeholder="e.g., I like quiet places with parks and good cafes. I dislike heavy traffic.",
            key="preferences_input"
        )
        st.info("After setting your preferences, ask me 'What do you think about Koramangala?' to get a match score!")
        
        st.divider()

        # --- NEW FEATURE: Clear Chat Button ---
        st.header("Chat Controls")
        if st.button("🗑️ Clear Chat History"):
            # Clear the messages from the session state
            st.session_state.messages = [
                AIMessage(content="Hello! How can I help you find your next home in Bangalore today?")
            ]
            # Rerun the app to reflect the changes immediately
            st.rerun()


    # --- Load Models and Vector Store (Cached for performance) ---
    @st.cache_resource
    def load_resources():
        """Load all the necessary models and data once."""
        try:
            embeddings = get_huggingface_embeddings()
            # This now loads the index if it exists, or creates it if it doesn't.
            vector_store = get_or_create_vector_store("data", embeddings)
            chat_model = get_chatgroq_model()
            return vector_store, chat_model
        except Exception as e:
            st.error(f"Error loading resources: {e}")
            return None, None

    vector_store, chat_model = load_resources()

    if not chat_model:
        st.warning("Could not initialize the chat model. Please check your API keys and configuration.")
        return

    # --- Chat History Management ---
    if "messages" not in st.session_state:
        st.session_state.messages = [
            AIMessage(content="Hello! How can I help you find your next home in Bangalore today?")
        ]

    # Display chat messages
    for message in st.session_state.messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

    # --- Handle User Input ---
    if prompt := st.chat_input("Ask me about rentals, or type /shortlist for a summary..."):
        # Add user message to history and display it
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- NEW FEATURE: Conversation Summary & Property Shortlist ---
        if prompt.strip().lower() == '/shortlist':
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your conversation to create a shortlist..."):
                    chat_history_str = format_chat_history(st.session_state.messages)
                    
                    summary_prompt = f"""
                    You are a helpful assistant. Review the following conversation history between a user and an AI rental assistant.
                    Your task is to identify the specific properties (using their IDs from the CSV if mentioned) that the user showed positive interest in.
                    Present these properties as a clean, bulleted list for the user, including their area, rent, and BHK.
                    Ignore any properties the user dismissed or was not interested in. If no specific properties were liked, state that.

                    Conversation History:
                    ---
                    {chat_history_str}
                    ---
                    """
                    
                    try:
                        summary_response = chat_model.invoke([HumanMessage(content=summary_prompt)])
                        response_content = summary_response.content
                    except Exception as e:
                        response_content = f"Sorry, I encountered an error while creating your shortlist: {e}"

                    st.markdown(response_content)
                    st.session_state.messages.append(AIMessage(content=response_content))

        # --- Standard Chat Logic ---
        else:
            with st.chat_message("assistant"):
                with st.spinner("Searching for the perfect place..."):
                    # 1. Get context from RAG
                    rag_context = get_context_from_rag(vector_store, prompt) if vector_store else "No local knowledge base found."

                    # 2. Get context from Web Search
                    search_context = perform_web_search(f"rental properties Bangalore {prompt}")

                    # 3. Construct a detailed system prompt with all context
                    system_prompt = f"""
                    You are an expert AI real estate assistant for Bangalore. Your goal is to help the user find a rental property.
                    You must use the provided context to answer the user's query.

                    **Response Mode:** Please provide a {response_mode} response.

                    **Context from my Knowledge Base (Neighborhood Info & Listings):**
                    ---
                    {rag_context}
                    ---

                    **Context from a Live Web Search (Current Info):**
                    ---
                    {search_context}
                    ---

                    Based on all the above information, answer the user's query.
                    Synthesize the information from the knowledge base and the web search into a single, helpful, conversational response.
                    If you use information from the web, mention that it's from a "real-time search."
                    """

                    try:
                        # Prepare messages for the model
                        formatted_messages = [SystemMessage(content=system_prompt)]
                        formatted_messages.extend(st.session_state.messages)

                        # Get response from model
                        response = chat_model.invoke(formatted_messages)
                        response_content = response.content

                        # --- NEW FEATURE: Check for Neighborhood Match Score ---
                        if st.session_state.user_preferences and any(area.lower() in prompt.lower() for area in ["koramangala", "hsr layout", "indiranagar", "jayanagar", "whitefield", "marathahalli", "electronic city", "jp nagar", "majestic", "nagasandra"]):
                            # Find which area was mentioned
                            mentioned_area = next((area for area in ["koramangala", "hsr layout", "indiranagar", "jayanagar", "whitefield", "marathahalli", "electronic city", "jp nagar", "majestic", "nagasandra"] if area.lower() in prompt.lower()), None)
                            
                            if mentioned_area:
                                with st.spinner(f"Calculating match score for {mentioned_area.title()}..."):
                                    try:
                                        # Load neighborhood description
                                        with open(f"data/{mentioned_area.replace(' ', '_')}.txt", "r") as f:
                                            neighborhood_desc = f.read()

                                        score_prompt = f"""
                                        Based on the user's preferences and the neighborhood description, provide a match score.
                                        User Preferences: "{st.session_state.user_preferences}"
                                        Neighborhood Description for {mentioned_area.title()}: "{neighborhood_desc}"
                                        
                                        On a scale of 1 to 10, how well does this neighborhood match the user's preferences?
                                        Provide the score and a single, concise sentence explaining your reasoning.
                                        Format your response as: **Match Score for {mentioned_area.title()}: [Score]/10** \n [Your reasoning].
                                        """
                                        score_response = chat_model.invoke([HumanMessage(content=score_prompt)])
                                        # Append the score to the main response
                                        response_content += "\n\n---\n\n" + score_response.content
                                    except Exception as e:
                                        response_content += f"\n\n---\n\nCould not calculate match score due to an error: {e}"

                    except Exception as e:
                        response_content = f"Sorry, I encountered an error: {e}"

                    st.markdown(response_content)
                    st.session_state.messages.append(AIMessage(content=response_content))

# --- Main App Execution ---
def main():
    st.set_page_config(
        page_title="AI Rental Assistant",
        page_icon="🏡",
        layout="wide"
    )
    chat_page()

if __name__ == "__main__":
    main()
