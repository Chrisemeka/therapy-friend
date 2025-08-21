import express, { Express, Request, Response } from 'express';
import dotenv from 'dotenv';
import { checkBibleVectorDB, checkTherapyVectorDB } from "./data_ingestion/check_vectordb";
import { BibleIngestion } from "./data_ingestion/bible_ingestion";
import { TherapyDataIngestion } from "./data_ingestion/therapy_practice_ingestion";

dotenv.config();

const app: Express = express();
const port = process.env.PORT || 5000;

app.use(express.json());

async function initializeService() {
    try {
        console.log("Starting the RAG pipeline...");

        console.log("Checking vector DB status...");
        const bibleHasData = await checkBibleVectorDB();
        const therapyHasData = await checkTherapyVectorDB();

        if (!bibleHasData) {
            console.log("No data found in bible vector DB. Ingesting data...");
            const ingestion = new BibleIngestion();
            await ingestion.ingest_data();

            console.log('Bible vector database populated successfully');
        }
        else {
            console.log('Bible vector database already contains data, skipping upload');
        }

        if (!therapyHasData) {
            console.log("No data found in therapy vector DB. Ingesting data...");
            const ingestion = new TherapyDataIngestion();
            await ingestion.ingest_data();

            console.log('Therapy vector database populated successfully');
        }
        else {
            console.log('Therapy vector database already contains data, skipping upload');
        }

        app.listen(port, () => {
            console.log(`The RAG pipeline is running on port http://localhost:${process.env.PORT}`);
        });
    } catch (error) {
        console.error('Failed to initialize service:', error);
        process.exit(1);
    }
}

initializeService();
