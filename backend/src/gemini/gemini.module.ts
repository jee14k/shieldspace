import { Module } from '@nestjs/common';
import { AiEmotionsController } from './gemini.controller';
import { AiEmotionsService } from './gemini.service';
import { JwtService } from '@nestjs/jwt';

@Module({
  providers: [AiEmotionsService, JwtService],
  controllers: [AiEmotionsController],
})
export class AiEmotionsModule {}
