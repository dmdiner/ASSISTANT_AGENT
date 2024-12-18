import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults

# Page Header
st.set_page_config(page_title="Movie Recommender")
st.title("Movie Recommender")
st.markdown("Movie Recommender Powered by Groq.")
st.markdown("### Help casual movie goers find possible movies.")

# Model and Agent Tools
llm = ChatGroq(api_key=st.secrets["GROQ_API_KEY"], model="llama3-8b-8192")
parser = StrOutputParser()
search = TavilySearchResults(max_results=2)

prompt_template = ChatPromptTemplate([("system", "You're an expert at text summarizing"), ("user", "Summarize this text: {text}")])

# Output Parser
parser = StrOutputParser()



# Store text to summarize in session state
st.session_state["summarized_text"] = ""

# User Interface
with st.form("movie_insight", clear_on_submit=False):
    # text to summarize
    movie_genre = st.text_input("Movie Genre:")
    movie_similar = st.text_area("Similar Movies To:")
    #company_url = st.text_input("Company URL:")
    #product_category = st.text_input("Product Category:")
    #competitors_urls = st.text_area("Competitors URL list, one per line:")
    #value_proposition = st.text_input("Value Proposition:")
    #target_customer = st.text_input("Target Customer:")


    # For the llm insight result
    movie_insights = ""

    # Data process
    if st.form_submit_button("Generate Insights"):
        if movie_genre:
            st.spinner("Processing...")

            # Internet search
            movie_search = "IMDB " + movie_genre
            movie_data = search.invoke(movie_search)
            print(movie_data)

            prompt = """"
            You are an expert movie connoisseur helping guests find movies they would be interested in watching, given the genre they request, and similar to any
            movies they provide. You are to give some themes from the movie you are recommending, but you are to avoid spoiling any plot of the movie in doing so.

            movie_genre: {movie_genre}
            movie_data: {movie_data}
            movie_similar: {movie_similar}

            Can you please create a list of movies, that the consumer may be interested in seeing, based off of
            their genre choice. If provided, also make sure the movies found are similar to the list of movies from movie_similar.
            Please return the data in this format. Don't include any URLs or images. Can you please have each criteria on their own line, and an empty line between
            each movie, and have the line with the movie title be bolded.
            Movie Title: ### (Year released)
            \nIMDB Rating: #.#/10
            \nLength: ### minutes
            \nGenre: ###, ###
            \nSimilarities: ###
            """
            #prompt_template = ChatPromptTemplate([("system", "You are an expert at finding business insight, get data for this user and their competitors"), 
                                    #("user", "Give me insight on this business and their competitors {movie_data}")])
            
            prompt_template = ChatPromptTemplate([("system", prompt)])

            # Chain
            chain = prompt_template | llm | parser

            # Results
            movie_insights = chain.invoke({"movie_data": movie_data,
                                             "movie_genre": movie_genre,
                                             "movie_similar": movie_similar})

    st.markdown(movie_insights)
