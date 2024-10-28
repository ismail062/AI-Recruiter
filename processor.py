import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

from dotenv import load_dotenv

load_dotenv()

class Processor:
    def __init__(self) -> None:
        self.model= ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
            )
        self.json_parser = JsonOutputParser()
        
    def extractJob(self, description):
        prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
        {page_data}
        ### INSTRUCTION:
        The scraped text is from the career's page of a website.
        Your job is to extract the job postings and return them in JSON format containing the 
        following keys: `role`, `experience`, `skills`, `tools` and `description`. Skills and tools should be list
        Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE):    
        """)

        chain_extract = prompt_extract | self.model 
        res = chain_extract.invoke(input={'page_data':description})
        try:
            job_response = self.json_parser.parse(res.content)
        except OutputParserException:
             raise OutputParserException("Context too big. Unable to parse jobs.")
        return job_response if isinstance(job_response, list) else job_response
    
    def getProfile(self, profile):
        prompt_cv_extract = PromptTemplate.from_template(
        """
        ### ANALYSE THE TEXT:
        {cv_text}
        ### INSTRUCTION:
        you have given a cv text.
        Your job is to extract the information and return them in JSON format containing the 
        following keys: `Name`, `email`, `contacts`, `role`, `experience`, `skills`, `tools` and `description`. 
        Summarize the CV and count total experience in number of years
        Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE):    
        """)

        chain_extract = prompt_cv_extract | self.model 
        res = chain_extract.invoke(input={'cv_text':profile})
        try:
            candidate_response = self.json_parser.parse(res.content)
        except OutputParserException:
             raise OutputParserException("Context too big. Unable to parse jobs.")
        return candidate_response
    

    def matchProfile(self, description, profile ):
        
        candidate_response = self.getProfile(profile=profile)

        prompt_email = PromptTemplate.from_template(
        """
        ### JOB DESCRIPTION:
        {job_description}

        ### Candidate profile
        {candidate_response}
        
        ### INSTRUCTION:4
       you are a hiring manager and expert in recruiting people. Your job is to match the {cv_text} with {job_description}.
       you can match the {candidate_response} and write the overall score of skill matches with the job_description. 
       write an email to given candidate email and explain the skills scores and invite for interview if skills matches more than 50%.
       If skills do not match then decline for the interview.  

      
        Do not provide a preamble.
        ### EMAIL (NO PREAMBLE):
        
        """
        )

        selection_chain_email = prompt_email | self.model
        selection_response = selection_chain_email.invoke({"job_description": str(description), "cv_text": profile, "candidate_response": candidate_response })
        return selection_response.content
    
    if __name__ == "__main__":
        print(os.getenv("GROQ_API_KEY"))
