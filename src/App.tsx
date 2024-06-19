/* eslint-disable @typescript-eslint/no-explicit-any */
import { useState, FormEvent, ChangeEvent } from "react";
import { Button, Input, List, Layout, Form as AntForm, Typography } from "antd";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import {
  RunnableSequence,
  RunnablePassthrough,
} from "@langchain/core/runnables";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { retriever } from "./utils/retriever";
import { combineDocuments } from "./utils/combineDocuments";
import { formatConvHistory } from "./utils/formatConvHistory";
import { uploadDocuments } from "./utils/upload_kb";

const { Content } = Layout;
const { TextArea } = Input;
const { Title } = Typography;

const googleAIApiKey = import.meta.env.VITE_GOOGLE_API_KEY;
const llm = new ChatGoogleGenerativeAI({ apiKey: googleAIApiKey });

const standaloneQuestionTemplate = `Given some conversation history (if any) and a question, convert the question to a standalone question. 
conversation history: {conv_history}
question: {question} 
standalone question:`;
const standaloneQuestionPrompt = PromptTemplate.fromTemplate(
  standaloneQuestionTemplate
);

const answerTemplate = `You are a helpful and enthusiastic support bot who can answer a given question about Samss based on the context provided and the conversation history. Try to find the answer in the context. If the answer is not given in the context, find the answer in the conversation history if possible. If you really don't know the answer, say "I'm sorry, I don't know the answer to that." And direct the questioner to email samssjubair@gmail.com. Don't try to make up an answer. Always speak as if you were chatting to a friend.
context: {context}
conversation history: {conv_history}
question: {question}
answer: `;
const answerPrompt = PromptTemplate.fromTemplate(answerTemplate);

const standaloneQuestionChain = standaloneQuestionPrompt
  .pipe(llm)
  .pipe(new StringOutputParser());

const retrieverChain = RunnableSequence.from([
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (prevResult: any) => prevResult.standalone_question,
  retriever,
  combineDocuments,
]);

const answerChain = answerPrompt.pipe(llm).pipe(new StringOutputParser());

const chain = RunnableSequence.from([
  {
    standalone_question: standaloneQuestionChain,
    original_input: new RunnablePassthrough(),
  },
  {
    context: retrieverChain,
    question: ({ original_input }: any) => original_input.question,
    conv_history: ({ original_input }: any) => original_input.conv_history,
  },
  answerChain,
]);

type ConversationEntry = {
  type: "human" | "ai";
  text: string;
};

const App = () => {
  const handleButtonClick = async () => {
    try {
      await uploadDocuments();
      console.log("Documents uploaded successfully.");
    } catch (error) {
      console.error("Failed to upload documents:", error);
    }
  };
  const [convHistory, setConvHistory] = useState<string[]>([]);
  const [question, setQuestion] = useState<string>("");
  const [conversation, setConversation] = useState<ConversationEntry[]>([]);

  const handleSubmit = async () => {
    // e.preventDefault();
    const userInput = question;
    setQuestion("");

    // Add human message
    setConversation((prevConversation) => [
      ...prevConversation,
      { type: "human", text: userInput },
    ]);

    const response = await chain.invoke({
      question: userInput,
      conv_history: formatConvHistory(convHistory),
    });

    setConvHistory((prevHistory) => [...prevHistory, userInput, response]);

    // Add AI message
    setConversation((prevConversation) => [
      ...prevConversation,
      { type: "ai", text: response },
    ]);
  };

  return (
    <Layout style={{ height: "100vh", padding: "20px" }}>
      <Content
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
        }}
      >
        <Button
          type="primary"
          onClick={handleButtonClick}
          style={{ marginBottom: "20px" }}
        >
          Upload Documents
        </Button>

        <Title level={2}>Chat with S-Bot</Title>

        <List
          style={{
            width: "100%",
            maxWidth: "600px",
            height: "400px",
            overflowY: "scroll",
            marginBottom: "20px",
            backgroundColor: "#f0f2f5",
            padding: "10px",
            borderRadius: "8px",
          }}
          dataSource={conversation}
          renderItem={(entry) => (
            <List.Item
              style={{
                justifyContent:
                  entry.type === "human" ? "flex-start" : "flex-end",
              }}
            >
              <div
                style={{
                  padding: "10px 15px",
                  borderRadius: "20px",
                  maxWidth: "70%",
                  backgroundColor:
                    entry.type === "human" ? "#0084ff" : "#e4e6eb",
                  color: entry.type === "human" ? "#fff" : "#000",
                  textAlign: entry.type === "human" ? "left" : "right",
                }}
              >
                {entry.text}
              </div>
            </List.Item>
          )}
        />

        <AntForm
          onFinish={handleSubmit}
          style={{ width: "100%", maxWidth: "600px" }}
        >
          <AntForm.Item>
            <TextArea
              rows={4}
              value={question}
              onChange={(e: ChangeEvent<HTMLTextAreaElement>) =>
                setQuestion(e.target.value)
              }
              placeholder="Ask a question..."
            />
          </AntForm.Item>
          <AntForm.Item>
            <Button type="primary" htmlType="submit" block>
              Send
            </Button>
          </AntForm.Item>
        </AntForm>
      </Content>
    </Layout>
  );
};

export default App;
