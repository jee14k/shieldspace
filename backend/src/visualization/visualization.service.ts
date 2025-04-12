import { Injectable } from '@nestjs/common';
import { DataSource } from 'typeorm';

export type VisualizationResponse = {
  label: string;
  'count(label)': string;
};

@Injectable()
export class VisualizationService {
  constructor(private dataSource: DataSource) {}

  async getBullyingCount(): Promise<VisualizationResponse[]> {
    const query = `
      select label, count(label) from shieldspace.bully_message where label != "not_cyberbullying" group by label;
    `;
    const result = await this.dataSource.query(query);
    return result;
  }
}
