export interface KeyFacts {
  type: string;
  height?: string;
  topSpeed?: string;
  duration: string;
  inversions?: string;
  capacity?: string;
  min_height?: string;
  min_age?: string;
  track_length?: string;
  manufacturer?: string;
  opening_year?: string;
  location?: string;
  'g-force'?: string;
}

export interface Attraction {
  id: string;
  name: string;
  short_description: string;
  description: string;
  keyFacts: KeyFacts;
  category: 'thrill' | 'family' | 'children' | 'water' | 'interactive';
  waitTime: number | null;
  predictedWaitTime: number | null;
  imageUrl: string;
  area: string;
}

export interface WeatherCondition {
  temperature: number;
  condition: string;
  icon: string;
}

export interface ForecastItem {
  time: number;
  temperature: number;
  condition: string;
}

export interface Weather {
  current: WeatherCondition;
  forecast: ForecastItem[];
} 