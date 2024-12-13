import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults

# Page Header
st.set_page_config(page_title="Assistant Agent")
st.title("Assistant Agent")
st.markdown("Assistant Agent Powered by Groq.")
st.markdown("### Help researchers gather insights from academic papers, extract summaries, and identify key references.")

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
with st.form("company_info", clear_on_submit=True):
    # text to summarize
    product_name = st.text_input("Product Name:")
    company_url = st.text_input("Company URL:")
    product_category = st.text_input("Product Category:")
    competitors_urls = st.text_area("Competitors URL list, one per line:")
    value_proposition = st.text_input("Value Proposition:")
    target_customer = st.text_input("Target Customer:")


    # For the llm insight result
    company_insights = ""

    # Data process
    if st.form_submit_button("Generate Insights"):
        if product_name and company_url:
            st.spinner("Processing...")

            # Internet search
            company_data = search.invoke(company_url)
            print(company_data)

            prompt = """"
            You are an expert customer service representative helping a potential customer comprehend the value of our product.
            Your goal is to convince the customer that our product has value, while addressing what the customer needs.

            Company Data: {company_data}
            Product name: {product_name}
            company_data: {company_data}
            product_name: {product_name}
            product_category: {product_category}
            competitors_urls: {competitors_urls}
            value_proposition: {value_proposition}
            target_customer: {target_customer}

            Can you please generate a report listing the following:
            - The company's strategy, summarize what it does in the industry relevent to the product they are selling.
            - List anypublic statements or press releases that are discussing the topic.
            - If the company has mentioned the Competitor, and if so, how the competitor is relevent to the company.
            - List any key leaders of the company.
            - Provide some information on the product/strategy of the company, include insight from 10-Ks and annual reports, if available.
            - Link articles or press releases that are relevent or mentioned in the report generated

            After the report, pelase create a short text to email the Target Customer to relay some information to convince them to purchase this product.        
            """
            #prompt_template = ChatPromptTemplate([("system", "You are an expert at finding business insight, get data for this user and their competitors"), 
                                    #("user", "Give me insight on this business and their competitors {company_data}")])
            
            prompt_template = ChatPromptTemplate([("system", prompt)])

            # Chain
            chain = prompt_template | llm | parser

            # Results
            company_insights = chain.invoke({"company_data": company_data,
                                             "product_name": product_name,
                                             "product_category": product_category,
                                             "competitors_urls": competitors_urls,
                                             "value_proposition": value_proposition,
                                             "target_customer": target_customer})

    st.markdown(company_insights)
