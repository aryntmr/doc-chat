# Doc Chat Bot

Doc Chat Bot is a Streamlit-based application that allows users to upload PDF documents and chat with a conversational AI to retrieve answers and insights from the uploaded documents. The application leverages LangChain for managing document embeddings and OpenAI's GPT models for answering user queries.

---

## Features
- **File Upload**: Users can upload a PDF document to the application.
- **Document Parsing**: The app processes the document and splits it into manageable chunks for efficient querying.
- **Conversational Interface**: Users can ask questions about the document and receive accurate answers.
- **Dynamic Document Embedding**: Embeddings are generated dynamically for each uploaded document using sentence transformers.
- **Powered by OpenAI**: Uses OpenAI GPT models for intelligent conversational responses.

---

## Repository Link

The project repository can be accessed at [Doc Chat Bot Repository](https://github.com/aryntmr/doc-chat).

## Installation

### Prerequisites
1. Python 3.8 or higher.
2. An OpenAI API key. Store this securely in the environment variables or Streamlit secrets.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name/doc-chat-bot.git
   cd doc-chat-bot
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file and add your OpenAI API key:
     ```plaintext
     OPENAI_API_KEY=your_openai_api_key
     ```
   - Alternatively, configure the API key in Streamlit secrets if deploying on Streamlit Cloud.

4. Run the application locally:
   ```bash
   streamlit run chat_ui.py
   ```

---

## File Structure
```
.
├── chat_ui.py          # Main Streamlit application interface
├── main_embedding.py   # Backend logic for document embedding and querying
├── requirements.txt    # List of dependencies
```

---

## Usage
1. Start the app using `streamlit run chat_ui.py`.
2. Upload a PDF document via the file uploader in the app.
3. Ask questions about the document in the chat interface.
4. Receive answers extracted from the uploaded document or general knowledge if needed.

---

## Deployment

### Deploying on Streamlit Cloud
1. Push your code to a public GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://streamlit.io/cloud) and create a new app.
3. Select your repository, branch, and the `chat_ui.py` file as the entry point.
4. Add your OpenAI API key to the **Secrets** section in the app settings.
5. Deploy and test your app.

### Live App
You can access the live app at [Doc Chat Bot](https://doc-chat-c2jkgocfvtshgsc8vpsuet.streamlit.app).
---

## Dependencies
The application relies on the following key libraries:
- `streamlit`: For building the interactive web interface.
- `langchain`: For managing document embeddings and retrieval chains.
- `faiss-cpu`: For efficient similarity search on embeddings.
- `sentence-transformers`: For generating high-quality embeddings.
- `pymupdf`: For parsing PDF files.
- `pdfminer.six`: For additional PDF processing capabilities.
- `python-dotenv`: For managing environment variables.
- `openai`: For accessing OpenAI GPT models.

---

## Contributing
If you would like to contribute:
1. Fork the repository.
2. Create a new branch for your feature:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push them:
   ```bash
   git commit -m "Description of changes"
   git push origin feature-name
   ```
4. Open a pull request.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments
- [Streamlit](https://streamlit.io/) for the framework.
- [LangChain](https://www.langchain.com/) for document and language model integration.
- [OpenAI](https://openai.com/) for GPT models.

---

## Contact
For issues or questions, please open an issue in the GitHub repository or contact the maintainer at `your-email@example.com`.

