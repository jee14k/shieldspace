import { Body, Controller, Injectable, Post, UseGuards } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { AxiosResponse } from 'axios';
import { AuthGuard } from 'src/auth/auth.guard';

export type Emotions = {
  user_input: string;
};

@Controller('model')
export class EmotionsController {
  constructor(private readonly httpService: HttpService) {}

  @Post('emotions')
  async getEmo(@Body() userInput: Emotions): Promise<AxiosResponse<{}>> {
    const { data } = await this.httpService.axiosRef.post(
      'http://localhost:8000/emotions',
      {
        user_input: userInput.user_input,
      },
    );
    return data;
  }
}
