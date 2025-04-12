import { Body, Controller, Post, UseGuards } from '@nestjs/common';
import { AiEmotionsService } from './gemini.service';
import { AuthGuard } from 'src/auth/auth.guard';
import { Emotions } from 'src/emotions/emotions.controller';

@Controller()
export class AiEmotionsController {
  constructor(private readonly aiService: AiEmotionsService) {}

  @UseGuards(AuthGuard)
  @Post('emotions')
  async getAiEmo(@Body() userInput: Emotions): Promise<{}> {
    return this.aiService.getEmo(userInput.user_input);
  }
}
