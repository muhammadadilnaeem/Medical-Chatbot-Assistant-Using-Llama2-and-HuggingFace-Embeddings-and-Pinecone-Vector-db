
---

# **🩺 Medical Chatbot Assistant using LLaMA 2, Hugging Face, and Pinecone**

Welcome to the **Medical Chatbot Assistant** project! This repository contains a powerful and efficient medical chatbot built using the LLaMA 2 model, Hugging Face embeddings, and Pinecone vector database. The chatbot is designed to assist users with medical inquiries, providing reliable and accurate responses.

https://github.com/user-attachments/assets/5f284cde-532f-4c96-9258-1b2760d254ee


## **🚀 Features**

- **LLaMA 2 Model Integration**: Powered by Meta's LLaMA 2 model, offering state-of-the-art conversational AI.
- **Hugging Face Embeddings**: Utilizes Hugging Face's embeddings for precise and context-aware responses.
- **Pinecone Vector Database**: Efficiently stores and retrieves embeddings, ensuring quick and relevant answers.
- **Scalable**: Easily scale the system to handle a growing number of users and queries.
- **Customizable**: Adapt the chatbot for various medical specializations or integrate it with other healthcare systems.

## **🛠️ Installation**

1. **Clone the repository:**

   ```bash
   git clone https://github.com/muhammadadilnaeem/Medical-Chatbot-Assistant-Using-Llama2-and-HuggingFace-Embeddings-and-Pinecone-Vector-db.git
   cd Medical-Chatbot-Assistant-Using-Llama2-and-HuggingFace-Embeddings-and-Pinecone-Vector-db
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**

   Create a `.env` file in the root directory and add your API keys and configuration settings:

   ```env
   HUGGINGFACE_API_KEY=your_huggingface_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

4. **Run the application:**

   ```bash
   python app.py
   ```

## **📚 Usage**

- **Ask Medical Questions**: The chatbot is trained to understand and respond to a wide range of medical queries. Simply type your question, and the bot will provide an accurate response.
- **Customize the Knowledge Base**: You can add or modify the medical data the chatbot uses by updating the embeddings stored in Pinecone.

## **🧠 How It Works**

1. **User Query**: The user inputs a medical question.
2. **Embeddings**: The question is converted into embeddings using Hugging Face models.
3. **Pinecone Retrieval**: The embeddings are matched against a database of medical knowledge stored in Pinecone.
4. **Response Generation**: The LLaMA 2 model generates a response based on the retrieved information.

## **🤖 Future Enhancements**

- **Multi-language Support**: Extend the chatbot to support multiple languages.
- **Voice Interface**: Integrate with speech-to-text and text-to-speech for a more interactive experience.
- **Integration with EHR Systems**: Connect the chatbot to Electronic Health Records (EHR) for personalized advice.

## **📄 License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.


## **📧 Contact**

For any questions or inquiries, please reach out to me at [madilnaeem0@gmail.com](madilnaeem0@gmail.com).


