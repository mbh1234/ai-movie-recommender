"""
Drift Detection module for monitoring real-time data quality changes
"""

from collections import deque
import logging
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)

class DriftMonitor:
    """Monitors data drift using a rolling window of events"""
    
    def __init__(self, window_size: int = 100, drift_threshold: float = 0.30):
        """
        Initialize drift monitor
        
        Args:
            window_size: Size of the rolling event buffer
            drift_threshold: Threshold for drift detection (failure rate)
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.event_buffer = deque(maxlen=window_size)
        
    def add_event(self, event: Dict) -> Dict:
        """
        Add new event to the rolling buffer and check for drift
        
        Args:
            event: Event dictionary containing at least 'status' field
            
        Returns:
            Dictionary containing drift detection results
        """
        # Add event to buffer
        self.event_buffer.append(event)
        
        # Only check drift if buffer has enough events
        if len(self.event_buffer) >= self.window_size:
            return self.detect_drift()
        
        return {
            'drift_detected': False,
            'message': f"Insufficient events ({len(self.event_buffer)}/{self.window_size})"
        }
        
    def detect_drift(self) -> Dict:
        """
        Detect drift based on failure rate in current window
        
        Returns:
            Dictionary containing drift detection results
        """
        # Count failed events (status = 400)
        failed_events = sum(1 for e in self.event_buffer if e.get('status') == 400)
        
        # Calculate failure rate
        failure_rate = failed_events / len(self.event_buffer)
        
        # Detect drift if failure rate exceeds threshold
        drift_detected = failure_rate > self.drift_threshold
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': drift_detected,
            'failure_rate': failure_rate,
            'failed_events': failed_events,
            'total_events': len(self.event_buffer),
            'threshold': self.drift_threshold
        }
        
        if drift_detected:
            logger.warning(
                f"Data drift detected: {failure_rate:.1%} failure rate in last "
                f"{self.window_size} events (threshold: {self.drift_threshold:.1%})"
            )
        
        return result
