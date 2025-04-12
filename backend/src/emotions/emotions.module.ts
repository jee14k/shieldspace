import { Module } from '@nestjs/common';
import { HttpModule } from '@nestjs/axios';
import { EmotionsController } from './emotions.controller';
import { JwtService } from '@nestjs/jwt';

@Module({
  imports: [HttpModule],
  controllers: [EmotionsController],
  providers: [JwtService],
})
export class EmotionsModule {}
