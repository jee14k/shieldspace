import { Module } from '@nestjs/common';
import { EmotionsModule } from './emotions/emotions.module';
import { AiEmotionsModule } from './gemini/gemini.module';
import { AuthModule } from './auth/auth.module';
import { ConfigModule } from '@nestjs/config';
import { VisualizationModule } from './visualization/visualization.module';

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true }),
    EmotionsModule,
    AiEmotionsModule,
    AuthModule,
    VisualizationModule,
  ],
  controllers: [],
  providers: [],
})
export class AppModule {}
