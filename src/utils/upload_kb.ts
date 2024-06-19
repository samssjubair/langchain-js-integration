import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { createClient, SupabaseClient } from "@supabase/supabase-js";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { OpenAIEmbeddings } from "@langchain/openai";

export const uploadDocuments = async (): Promise<void> => {
  try {
    const result = await fetch("upload.txt");
    const text = await result.text();

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 50,
      separators: ["\n\n", "\n", " ", ""], // default setting
    });

    const output = await splitter.createDocuments([text]);

    const openAIApiKey: string = import.meta.env.VITE_OPENAI_API_KEY;
    const sbApiKey: string = import.meta.env.VITE_SUPABASE_API_KEY;
    const sbUrl: string = import.meta.env.VITE_SUPABASE_URL_LC_CHATBOT;

    const client: SupabaseClient = createClient(sbUrl, sbApiKey);

    await SupabaseVectorStore.fromDocuments(
      output,
      new OpenAIEmbeddings({ openAIApiKey }),
      {
        client,
        tableName: "documents",
      }
    );
  } catch (err) {
    console.error("Error uploading documents:", err);
  }
};
