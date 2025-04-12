import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { DataSource } from 'typeorm';
import { VisualizationController } from './visualization.controller';
import { VisualizationService } from './visualization.service';
import { ConfigModule } from '@nestjs/config';

@Module({
  imports: [
    ConfigModule.forRoot(),
    TypeOrmModule.forRoot({
      type: 'mysql',
      host: 'shieldspace.ck1sygqoe3hd.us-east-1.rds.amazonaws.com',
      port: 3306,
      username: process.env.DATABASE_USERNAME,
      password: process.env.DATABASE_PASSWORD,
      database: 'shieldspace',
      synchronize: false,
      autoLoadEntities: true,
    }),
  ],
  controllers: [VisualizationController],
  providers: [VisualizationService],
})
export class VisualizationModule {
  constructor(private dataSource: DataSource) {}
}
