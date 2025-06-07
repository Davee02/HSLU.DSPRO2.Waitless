// queueTimesService.ts

export interface QueueTimeData {
    id: number;
    name: string;
    is_open: boolean;
    wait_time: number;
    last_updated: string;
    landName?: string;
  }
  
  export interface QueueTimesResponse {
    lands: Array<{
      id: number;
      name: string;
      rides: QueueTimeData[];
    }>;
    rides: QueueTimeData[];
  }
  
  export interface RideWaitTime {
    localId: string;
    waitTime: number;
    isOpen: boolean;
    lastUpdated: string;
    apiId: number;
  }
  
  // Function to normalize ride names for comparison
  export const normalizeRideName = (name: string): string => {
    return name
      .toLowerCase()
      .trim()
      // Replace various types of dashes and hyphens with standard dash
      .replace(/[–—−]/g, '-')
      // Remove special characters and extra spaces
      .replace(/[''`´]/g, '')
      .replace(/[!]/g, '')
      .replace(/\s+/g, ' ')
      // Remove common prefixes/suffixes that might differ
      .replace(/\s*-\s*/g, '-')
      .replace(/^the\s+/i, '')
      .replace(/\s+powered\s+by\s+.*/i, '') // Remove "powered by X" suffixes
      .trim();
  };
  
  // Local ride name mappings (based on your attraction data)
  export const LOCAL_RIDE_MAPPINGS = [
    { id: "silver-star", name: "Silver Star" },
    { id: "blue-fire", name: "Blue Fire Megacoaster" },
    { id: "wodan", name: "Wodan – Timburcoaster" },
    { id: "voletarium", name: "Voletarium" },
    { id: "alpine-express-enzian", name: "Alpine Express Enzian" },
    { id: "arena-of-football", name: "Arena of Football - Be Part of It" },
    { id: "arthur", name: "Arthur" },
    { id: "atlantica-supersplash", name: "Atlantica SuperSplash" },
    { id: "atlantis-adventure", name: "Atlantis Adventure" },
    { id: "baaa-express", name: "Baaa Express" },
    { id: "bellevue-ferris-wheel", name: "Bellevue Ferris Wheel" },
    { id: "castello-dei-medici", name: "Castello dei Medici" },
    { id: "dancing-dingie", name: "Dancing Dingie" },
    { id: "euromir", name: "Euro-Mir" },
    { id: "eurosat-cancan-coaster", name: "Eurosat CanCan Coaster" },
    { id: "eurotower", name: "Euro-Tower" },
    { id: "fjordrafting", name: "Fjord-Rafting" },
    { id: "jim-button-journey", name: "Jim Button - Journey Through Morrowland" },
    { id: "josefinas-imperial-journey", name: "Josefina's Magical Imperial Journey" },
    { id: "kolumbusjolle", name: "Kolumbusjolle" },
    { id: "madame-freudenreich", name: "Madame Freudenreich Curiosités" },
    { id: "matterhorn-blitz", name: "Matterhorn-Blitz" },
    { id: "old-macdonald", name: "Old Mac Donald's Tractor Fun" },
    { id: "pegasus", name: "Pegasus" },
    { id: "pirates-in-batavia", name: "Pirates in Batavia" },
    { id: "poppy-towers", name: "Poppy Towers" },
    { id: "poseidon", name: "Poseidon" },
    { id: "snorri-touren", name: "Snorri Touren" },
    { id: "swiss-bob-run", name: "Swiss Bob Run" },
    { id: "tirol-log-flume", name: "Tirol Log Flume" },
    { id: "tnnevirvel", name: "Tnnevirvel" },
    { id: "vienna-wave-swing", name: "Vienna Wave Swing - Glckspilz" },
    { id: "vindjammer", name: "Vindjammer" },
    { id: "volo-da-vinci", name: "Volo da Vinci" },
    { id: "voltron-nevera", name: "Voltron Nevera - Powered by Rimac" },
    { id: "whale-adventures", name: "Whale Adventures - Northern Lights" }
  ];
  
  export const fetchQueueTimes = async (): Promise<Map<string, RideWaitTime>> => {
    try {
      // Use CORS proxy to bypass browser restrictions
      const proxyUrl = 'https://api.allorigins.win/get?url=';
      const targetUrl = 'https://queue-times.com/parks/51/queue_times.json';
      
      console.log('Fetching queue times via proxy...');
      const response = await fetch(`${proxyUrl}${encodeURIComponent(targetUrl)}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const proxyData = await response.json();
      
      // Check if the proxy request was successful
      if (proxyData.status.http_code !== 200) {
        throw new Error(`API returned status: ${proxyData.status.http_code}`);
      }
      
      // Parse the actual data from the proxy response
      const data: QueueTimesResponse = JSON.parse(proxyData.contents);
  
      // For debugging
      console.log('Raw queue times data:', data);
  
      // Create normalized lookup maps
      const localRideMap = new Map<string, { id: string; name: string }>();
      LOCAL_RIDE_MAPPINGS.forEach(ride => {
        localRideMap.set(normalizeRideName(ride.name), ride);
      });
  
      // Flatten API data structure
      const allAPIRides: QueueTimeData[] = [];
      if (data.lands) {
        data.lands.forEach(land => {
          if (land.rides) {
            land.rides.forEach(ride => {
              allAPIRides.push({
                ...ride,
                landName: land.name
              });
            });
          }
        });
      }
      if (data.rides) {
        allAPIRides.push(...data.rides);
      }
  
      // Match rides and create result map
      const resultMap = new Map<string, RideWaitTime>();
      
      allAPIRides.forEach(apiRide => {
        const normalizedAPIName = normalizeRideName(apiRide.name);
        const matchedLocal = localRideMap.get(normalizedAPIName);
        
        if (matchedLocal) {
          resultMap.set(matchedLocal.id, {
            localId: matchedLocal.id,
            waitTime: apiRide.wait_time,
            isOpen: apiRide.is_open,
            lastUpdated: apiRide.last_updated,
            apiId: apiRide.id
          });
        }
      });
  
      // For debugging
      console.log('Matched queue times:', Array.from(resultMap.entries()));
      console.log(`Successfully matched ${resultMap.size} rides`);
  
      return resultMap;
    } catch (error) {
      console.error('Error fetching queue times:', error);
      
      // Provide more specific error information
      if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
        throw new Error('Network error: Unable to reach the queue times service');
      } else if (error instanceof SyntaxError) {
        throw new Error('Invalid response format from queue times service');
      } else {
        throw error;
      }
    }
  };