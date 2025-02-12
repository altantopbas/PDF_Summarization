import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from pypdf import PdfReader
from dotenv import load_dotenv
import os
import google.generativeai as genai
# Load environment variables
load_dotenv()

# Configure Google Generative AI
#genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
def extract_text_from_pdf(pdf_file):
    """PDF dosyasından metin çıkarma"""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def create_summary_chain():
    """Özet çıkarma zinciri oluşturma"""
    llm = ChatGoogleGenerativeAI(
        model = "gemini-2.0-flash-exp",
        temperature=0.3,
        convert_system_message_to_human=True,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    summary_template = """
    Summarize the following text, heading by heading.
        Create a comprehensive summary that includes key points, main ideas, and key findings.
        The summary should be clear and concise.
        First summarize the text under one heading, then summarize the text under the next heading, and so on.
        Text:
        {text} 

       Summary:
    """

    summary_prompt = PromptTemplate(
        input_variables=["text"],
        template=summary_template
    )

    return LLMChain(llm=llm, prompt=summary_prompt)


def main():
    st.set_page_config(page_title="PDF Özetleyici", page_icon="📚")

    st.title("📚 PDF Özetleyici")
    st.write("PDF dosyanızı yükleyin ve yapay zeka destekli özetini alın.")

    uploaded_file = st.file_uploader("PDF dosyanızı seçin", type="pdf")

    if uploaded_file is not None:
        with st.spinner("PDF içeriği okunuyor..."):
            text = extract_text_from_pdf(uploaded_file)

        st.success("PDF başarıyla okundu!")

        if st.button("Özet Oluştur"):
            with st.spinner("Özet oluşturuluyor..."):
                # Özet zincirini oluştur
                summary_chain = create_summary_chain()

                # Metni bölümlere ayır (token limitini aşmamak için)
                max_chunk_size = 12000  # Yaklaşık 12K karakter
                text_chunks = [text[i:i + max_chunk_size]
                               for i in range(0, len(text), max_chunk_size)]

                # Her bölüm için özet oluştur
                summaries = []
                for chunk in text_chunks:
                    summary = summary_chain.invoke({"text": chunk})
                    summaries.append(summary["text"])

                # Tüm özetleri birleştir
                final_summary = "\n\n".join(summaries)

                st.subheader("📝 Özet")
                st.write(final_summary)

                # Özeti indirme seçeneği
                st.download_button(
                    label="Özeti İndir",
                    data=final_summary,
                    file_name="ozet.txt",
                    mime="text/plain"
                )


if __name__ == "__main__":
    main()




"""
Aşağıdaki metni başlık başlık özetle. 
    Önemli noktaları, ana fikirleri ve temel bulguları içeren kapsamlı bir özet oluştur.
    Özet anlaşılır ve akıcı olmalı. 
    Öncelikle bir başlığın altındaki yazıyı özetle, daha sonra diğer başlığın altındaki yazıyı özetle ve bu şekilde devam et.

    Metin:
    {text}

    Özet:
"""
    #"""
   # Summarize the following text, heading by heading.
   #     Create a comprehensive summary that includes key points, main ideas, and key findings.
   #     The summary should be clear and concise.
   #     First summarize the text under one heading, then summarize the text under the next heading, and so on.
   #     Text:
   #     {text}

   #     Summary:
    #"""