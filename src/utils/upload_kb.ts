import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { createClient, SupabaseClient } from "@supabase/supabase-js";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import textFile from "./upload.txt";

export const uploadDocuments = async (): Promise<void> => {
  try {
    const result = await fetch(textFile);
    const text = await result.text();

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 50,
      separators: ["\n\n", "\n", " ", ""], // default setting
    });

    const output = await splitter.createDocuments([text]);

    // const openAIApiKey: string = import.meta.env.VITE_OPENAI_API_KEY;
    const genAIApiKey: string = import.meta.env.VITE_GOOGLE_API_KEY;
    const sbApiKey: string = import.meta.env.VITE_SUPABASE_API_KEY;
    const sbUrl: string = import.meta.env.VITE_SUPABASE_URL_LC_CHATBOT;
    // console.log("sbUrl", sbUrl, sbApiKey);

    const client: SupabaseClient = createClient(sbUrl, sbApiKey);

    await SupabaseVectorStore.fromDocuments(
      output,
      new GoogleGenerativeAIEmbeddings({
        apiKey: genAIApiKey,
        model: "embedding-001",
      }),
      {
        client,
        tableName: "documents",
      }
    );
  } catch (err) {
    console.error("Error uploading documents:", err);
  }
};
