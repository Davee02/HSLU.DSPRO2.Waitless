export interface KeyFacts {
  type: string;
  height?: string;
  topSpeed?: string;
  duration: string;
  inversions?: string;
  capacity?: string;
}

export interface Attraction {
  id: string;
  name: string;
  description: string;
  keyFacts: KeyFacts;
  category: 'thrill' | 'family' | 'kids';
  waitTime: number | null;
  predictedWaitTime: number | null;
  imageUrl: string;
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