/* eslint-disable @typescript-eslint/no-explicit-any */
import  { useState, FormEvent, ChangeEvent } from "react";
import styled from "styled-components";
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

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
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
      { type: "human", text: userInput },
      { type: "ai", text: response },
    ]);
  };
  return (
    <Container>
      <div>
        <button onClick={handleButtonClick}>Upload Documents</button>
      </div>

      <ConversationContainer id="chatbot-conversation-container">
        {conversation.map((entry, index) => (
          <SpeechBubble key={index} className={`speech-${entry.type}`}>
            {entry.text}
          </SpeechBubble>
        ))}
      </ConversationContainer>
      <Form onSubmit={handleSubmit}>
        <Input
          type="text"
          id="user-input"
          value={question}
          onChange={(e: ChangeEvent<HTMLInputElement>) =>
            setQuestion(e.target.value)
          }
          placeholder="Ask a question..."
        />
        <Button type="submit">Send</Button>
      </Form>
    </Container>
  );
};

export default App;

// Styled Components
const Container = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  font-family: Arial, sans-serif;
`;

const ConversationContainer = styled.div`
  width: 80%;
  max-width: 600px;
  height: 400px;
  border: 1px solid #ccc;
  padding: 10px;
  overflow-y: scroll;
  margin-bottom: 20px;
`;

const SpeechBubble = styled.div`
  margin: 10px 0;
  padding: 10px;
  border-radius: 5px;
  background-color: #f1f1f1;

  &.speech-human {
    background-color: #e1ffc7;
    align-self: flex-start;
  }

  &.speech-ai {
    background-color: #c7eaff;
    align-self: flex-end;
  }
`;

const Form = styled.form`
  display: flex;
  width: 80%;
  max-width: 600px;
`;

const Input = styled.input`
  flex: 1;
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 5px;
  margin-right: 10px;
`;

const Button = styled.button`
  padding: 10px 20px;
  border: none;
  border-radius: 5px;
  background-color: #007bff;
  color: white;
  cursor: pointer;

  &:hover {
    background-color: #0056b3;
  }
`;
