import requests
import pandas as pd
import time
import csv
from datetime import datetime
from tqdm import tqdm
import random

# Configuration
BASE_URL = "http://localhost:8082"
NUM_REQUESTS = 2000
DELAY_MS = 600  # milliseconds between requests
OUTPUT_FILE = "load_test_results.csv"
USER_FILE = "unique_users.csv"

def load_users(filename):
    """Load user IDs from CSV file"""
    try:
        df = pd.read_csv(filename)
        # Try common column names
        if 'user_id' in df.columns:
            users = df['user_id'].tolist()
        elif 'userId' in df.columns:
            users = df['userId'].tolist()
        else:
            # Use first column if no standard name found
            users = df.iloc[:, 0].tolist()
        
        print(f"Loaded {len(users):,} users from {filename}")
        return users
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Using random user IDs.")
        return list(range(1, 100000))
    except Exception as e:
        print(f"Error loading users: {e}. Using random user IDs.")
        return list(range(1, 100000))

def send_request(user_id):
    """Send recommendation request and return results"""
    url = f"{BASE_URL}/recommend/{user_id}"
    
    try:
        start_time = time.time()
        response = requests.get(url, timeout=2)
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            'user_id': user_id,
            'status_code': response.status_code,
            'response': response.text[:500],  # Limit response length
            'response_time_ms': round(elapsed_ms, 2),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            'success': response.status_code == 200
        }
    except requests.exceptions.Timeout:
        return {
            'user_id': user_id,
            'status_code': 408,
            'response': 'Request Timeout',
            'response_time_ms': DELAY_MS * 2,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            'success': False
        }
    except requests.exceptions.ConnectionError:
        return {
            'user_id': user_id,
            'status_code': 503,
            'response': 'Connection Error',
            'response_time_ms': 0,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            'success': False
        }
    except Exception as e:
        return {
            'user_id': user_id,
            'status_code': 500,
            'response': str(e)[:200],
            'response_time_ms': 0,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            'success': False
        }

def main():
    print("="*80)
    print("SVD RECOMMENDATION SERVICE LOAD TEST")
    print("="*80)
    print(f"Target: {BASE_URL}")
    print(f"Requests: {NUM_REQUESTS:,}")
    print(f"Delay: {DELAY_MS}ms between requests")
    print(f"Output: {OUTPUT_FILE}")
    print("="*80)
    
    # Test connection first
    print("\nTesting connection...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print(f"✓ Service is running: {response.text}")
        else:
            print(f"✗ Service returned status {response.status_code}")
            return
    except Exception as e:
        print(f"✗ Cannot connect to service: {e}")
        print("Make sure the Docker container is running on port 8082")
        return
    
    # Load users
    print("\nLoading users...")
    users = load_users(USER_FILE)
    
    # Select random users for testing
    if len(users) > NUM_REQUESTS:
        test_users = random.sample(users, NUM_REQUESTS)
    else:
        test_users = random.choices(users, k=NUM_REQUESTS)
    
    print(f"Selected {len(test_users):,} user IDs for testing")
    
    # Initialize results storage
    results = []
    
    # Statistics
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'timeouts': 0,
        'total_time': 0,
        'min_time': float('inf'),
        'max_time': 0,
        'under_600ms': 0
    }
    
    # Open CSV file for writing
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'user_id', 'status_code', 'response_time_ms', 
                      'response', 'success']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        print(f"\nStarting load test...")
        print(f"Sending {NUM_REQUESTS:,} requests...\n")
        
        # Send requests with progress bar
        for user_id in tqdm(test_users, desc="Sending requests", unit="req"):
            # Send request
            result = send_request(user_id)
            
            # Write to CSV immediately
            writer.writerow(result)
            csvfile.flush()  # Ensure it's written to disk
            
            # Update statistics
            stats['total'] += 1
            if result['success']:
                stats['success'] += 1
            else:
                stats['failed'] += 1
            
            if result['status_code'] == 408:
                stats['timeouts'] += 1
            
            response_time = result['response_time_ms']
            if response_time > 0:
                stats['total_time'] += response_time
                stats['min_time'] = min(stats['min_time'], response_time)
                stats['max_time'] = max(stats['max_time'], response_time)
                
                if response_time < 600:
                    stats['under_600ms'] += 1
            
            # Store for analysis
            results.append(result)
            
            # Wait before next request (convert ms to seconds)
            time.sleep(DELAY_MS / 1000.0)
    
    # Calculate final statistics
    print("\n" + "="*80)
    print("LOAD TEST RESULTS")
    print("="*80)
    
    print(f"\nTotal Requests:    {stats['total']:,}")
    print(f"Successful (200):  {stats['success']:,} ({stats['success']/stats['total']*100:.1f}%)")
    print(f"Failed:            {stats['failed']:,} ({stats['failed']/stats['total']*100:.1f}%)")
    print(f"Timeouts:          {stats['timeouts']:,}")
    
    if stats['success'] > 0:
        avg_time = stats['total_time'] / stats['success']
        print(f"\nResponse Times:")
        print(f"  Average:         {avg_time:.2f}ms")
        print(f"  Min:             {stats['min_time']:.2f}ms")
        print(f"  Max:             {stats['max_time']:.2f}ms")
        print(f"  Under 600ms:     {stats['under_600ms']:,} ({stats['under_600ms']/stats['success']*100:.1f}%)")
        
        if avg_time < 600:
            print(f"\n✓ Average response time is under 600ms requirement!")
        else:
            print(f"\n✗ Average response time exceeds 600ms requirement")
    
    # Status code breakdown
    print(f"\nStatus Code Breakdown:")
    status_counts = {}
    for result in results:
        code = result['status_code']
        status_counts[code] = status_counts.get(code, 0) + 1
    
    for code, count in sorted(status_counts.items()):
        print(f"  {code}: {count:,} ({count/stats['total']*100:.1f}%)")
    
    # Sample responses
    print(f"\nSample Successful Responses:")
    success_results = [r for r in results if r['success']]
    if success_results:
        for i, result in enumerate(success_results[:3], 1):
            response_preview = result['response'][:80] + "..." if len(result['response']) > 80 else result['response']
            print(f"  {i}. User {result['user_id']}: {response_preview}")
    else:
        print("  No successful responses")
    
    print(f"\nResults saved to: {OUTPUT_FILE}")
    print("="*80)
    
    # Save summary
    summary_file = "load_test_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("SVD RECOMMENDATION SERVICE LOAD TEST SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Target: {BASE_URL}\n")
        f.write(f"Total Requests: {stats['total']:,}\n")
        f.write(f"Successful: {stats['success']:,} ({stats['success']/stats['total']*100:.1f}%)\n")
        f.write(f"Failed: {stats['failed']:,} ({stats['failed']/stats['total']*100:.1f}%)\n")
        f.write(f"Timeouts: {stats['timeouts']:,}\n\n")
        
        if stats['success'] > 0:
            f.write(f"Average Response Time: {stats['total_time']/stats['success']:.2f}ms\n")
            f.write(f"Min Response Time: {stats['min_time']:.2f}ms\n")
            f.write(f"Max Response Time: {stats['max_time']:.2f}ms\n")
            f.write(f"Under 600ms: {stats['under_600ms']:,} ({stats['under_600ms']/stats['success']*100:.1f}%)\n\n")
        
        f.write("Status Code Breakdown:\n")
        for code, count in sorted(status_counts.items()):
            f.write(f"  {code}: {count:,} ({count/stats['total']*100:.1f}%)\n")
    
    print(f"Summary saved to: {summary_file}\n")

if __name__ == "__main__":
    main()