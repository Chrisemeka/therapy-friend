import { PDFLoader  } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { Chroma } from "@langchain/community/vectorstores/chroma";

export class BibleIngestion {
    async ingest_data() {
        try {
           
            // 1. Loading the PDF file
            console.log("Loading Bible PDF file...");
            const pdfPath = "./data/kjv.pdf"; // PDF file path
            const loader = new PDFLoader(pdfPath);
            const docs = await loader.load();

            console.log(`PDF loaded successfully! Documents: ${docs.length}`);

            // Log first few characters to verify content
            if (docs.length > 0) {
                console.log("Sample content:", docs[0].pageContent.substring(0, 200));
            }

            // 2. Spilting the data into chunks
            console.log("Splitting data into chunks...");    
            const textSplitter = new RecursiveCharacterTextSplitter({
                chunkSize: 1000, // Maximum size of each chunk
                chunkOverlap: 100, // Overlap between consecutive chunks to maintain context
            });

            const splitDocs = await textSplitter.splitDocuments(docs);
            // 'splitDocs' will be an array of Document objects, each representing a chunk of the original PDF content.

            console.log(`Data split successfully! Chunks created: ${splitDocs.length}`);

            // 3. Initailize the embeddings model
            console.log("Initializing embeddings models...");
            const embeddings = new HuggingFaceTransformersEmbeddings({ 
                model: 'sentence-transformers/all-MiniLM-L6-v2' 
            })

            // 4. Create vector store and add documents
            console.log("Creating vector store and embedding documents...");
            const vectorStore = new Chroma(embeddings, {
                collectionName: "bible-collection",
            });
            const ids = splitDocs.map((_, index) => `doc-${index}`);
            await vectorStore.addDocuments(splitDocs, {ids});
            
            console.log("Data successfully embedded and stored in vector database!");
            return { success: true, documentsProcessed: splitDocs.length };

        } catch (error) {
            console.error("Error during ingestion:", error);
            throw error;
        }
    }

    async vector_search(query: string, topK = 10) {
        try {
            // 1. Generate embedding for the search query
            console.log("Generating embedding for query:", query);
            const embeddings = new HuggingFaceTransformersEmbeddings({ 
                model: 'sentence-transformers/all-MiniLM-L6-v2' 
            });
            
            const queryEmbedding = await embeddings.embedQuery(query);
            console.log("Query embedding generated, dimension:", queryEmbedding.length);

            //2. Retrieve the ChromaDb
            const vectorStore = new Chroma(embeddings, {
                collectionName: "bible-collection",
            });

            const results = await vectorStore.similaritySearch(query, topK);
            return results
        } catch (error) {
            console.error("Error during vector search:", error);
            throw error;
        }
    }
}

  