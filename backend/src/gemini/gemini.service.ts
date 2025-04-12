import { GenerateContentResponse, GoogleGenAI, Type } from '@google/genai';
import { Injectable } from '@nestjs/common';
import { response } from 'express';

const model = {
  model: 'gemini-2.5-pro-exp-03-25',
  systemInstruction:
    'You are a tone and emotion analysis system for online messages.\n' +
    'Given a comment, return tone probabilities (0 to 1), most likely emotions impacted, and bullying-related metadata in raw JSON format only — no explanation, no markdown wrapping.\n' +
    'Here is the format:\n' +
    '{ "tone_scores": ..., "bullying_likelihood": ... }\n' +
    'Here is the message:\n“{user_input}”',
};

const generationConfig = {
  temperature: 1,
  topP: 0.95,
  topK: 64,
  maxOutputTokens: 65536,
  responseMimeType: 'text/plain',
};

@Injectable()
export class AiEmotionsService {
  async getEmo(contents: string): Promise<string> {
    const apiKey = process.env.GEMINI_API_KEY;
    const ai = new GoogleGenAI({ apiKey });
    const { candidates } = await ai.models.generateContent({
      model: model.model,
      contents,
      config: {
        systemInstruction: model.systemInstruction,
        ...generationConfig,
      },
    });
    let response: string = '';

    candidates?.forEach((candidate) => {
      const { content } = candidate;
      if (content?.parts?.[0]?.text) {
        response = content.parts[0]?.['text'];
      }
    });

    if (!Object.entries(response).length) {
      throw new Error('No response from AI');
    }
    return response;
  }
}
