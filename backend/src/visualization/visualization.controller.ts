import { Controller, Get, UseGuards } from '@nestjs/common';
import {
  VisualizationResponse,
  VisualizationService,
} from './visualization.service';
import { AuthGuard } from 'src/auth/auth.guard';

@Controller()
@UseGuards(AuthGuard)
export class VisualizationController {
  constructor(private readonly visualizationService: VisualizationService) {}

  @Get('visualization')
  async getAllData(): Promise<VisualizationResponse[]> {
    return this.visualizationService.getBullyingCount();
  }
}
