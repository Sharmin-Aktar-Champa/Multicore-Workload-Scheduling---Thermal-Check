/*
Concurrent Dispatch Scheme:
Call concurrent function instead of pid_control
pthread_create(&pid_thread, NULL, concurrent, NULL);
*/

/*
Serialized Dispatch Scheme:
int big_core_id = 4; pick any big core 4, 5, 6, 7
pthread_create(&pid_thread, NULL, serialized, &big_core_id);
set_thread_affinity(pid_thread, 1);
*/

/*Hotspot Mitigation:
pthread_create(&pid_thread, NULL, hotspot_mitigation, NULL);
*/

/*Hybrid Single Dispatch:
pthread_create(&pid_thread, NULL, hybrid_single_dispatch, NULL);
*/

/*Hybrid Double Dispatch:
Initialize to the first two cores by default
Function to serialize threads across the two coolest cores
pthread_create(&pid_thread, NULL, hybrid_double_dispatch, NULL);
*/

/*Turning on off Fan programatically:
Register the signal handler for Ctrl+C
Turn off automatic fan control and set fan speed to 0
printf("Disabling automatic fan control and turning off the fan.\n");
write_sysfs(automatic_path, 0);  // Disable automatic mode
write_sysfs(pwm1_path, 0);  // Turn off the fan
Experiment or main program logic here
*/

/*Matrix Multiply:
Create a file name specific to the thread
Open the log file for writing
Write the header to the log file
Start timing the matrix multiplication
Stop timing the matrix multiplication
Calculate the latency for this operation
Log the latency to the file
Ensure the log file is updated
Close the log file when done
*/

/*
1 second = 1000 milliseconds
1 microsecond = 0.001 milliseconds
*/

#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/time.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include <iostream>
#include <sstream> 
#include <fstream>
#include <algorithm>
#include <vector>

#define NUM_CORES 4
#define TARGET_TEMP 50.0
#define MATRIX_SIZE 256
#define NUM_THREADS 4
#define LOG_INTERVAL (0.05*1000000) 
#define MONITOR_INTERVAL (0.05*1000000) //5 million microseconds, 1 second = 1,000,000 microseconds.
#define LOG_DURATION 1
#define NUM_CONFIGURATIONS (7*16*10*10*19*19)  // Total number of configurations (7 * 16 * 10 * 10 * 19 * 19)

// Define matrix and result structures
typedef struct {
    double A[MATRIX_SIZE][MATRIX_SIZE];
    double B[MATRIX_SIZE][MATRIX_SIZE];
    double C[MATRIX_SIZE][MATRIX_SIZE];
} matrix_data_t;

// Thread argument structure
typedef struct {
    int thread_id;
    matrix_data_t *data;
    pthread_t thread;
    int core_id;
} thread_arg_t;

typedef enum {
    CONCURRENT,
    SERIALIZED,
    REDUCED_FREQ,
    WAIT_FOR_THRESHOLD
} dispatch_state_t;

// Array to hold configurations (Tmin, Tmax, ds, dc, fs, fc)
typedef struct {
    int Tmin;
    int Tmax;
    int ds;
    int dc;
    int fs;
    int fc;
} Config;

// Arrays for UCB
double rewards[NUM_CONFIGURATIONS];
int counts[NUM_CONFIGURATIONS];
Config configurations[NUM_CONFIGURATIONS];
int selected_config;
int total_trials = 0;

pthread_t threads[NUM_THREADS];
pthread_t temp_thread;
thread_arg_t thread_args[NUM_THREADS];
matrix_data_t matrices[NUM_THREADS];
pthread_t pid_thread;
pthread_mutex_t lock; 
pthread_barrier_t barrier;  // Barrier declaration
double execution_times[NUM_THREADS];  // Array to store execution times of all threads

int MAX_PEAK_TEMP = 65.0;
int MIN_AVG_TEMP = 55.0;
int BIG_CORE_MIN_FREQ = 200000;    // 200000 KHz = 200 MHz
int BIG_CORE_MAX_FREQ = 2000000;   // 2000000 KHz = 2000 MHz
int serialized_duration = 5;       // Time in seconds for serialized mode
int concurrent_duration = 3;       // Time in seconds for concurrent mode

volatile int stop_logging = 0;
const char *big_core_paths[NUM_CORES] = {
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/thermal/thermal_zone1/temp",
        "/sys/class/thermal/thermal_zone2/temp",
        "/sys/class/thermal/thermal_zone3/temp"
    };

const char *automatic_path = "/sys/devices/platform/pwm-fan/hwmon/hwmon0/automatic";
const char *pwm1_path = "/sys/devices/platform/pwm-fan/hwmon/hwmon0/pwm1";

// const char * log_filename = "temperature_log_concurrent_v1.2.csv";
// const char * log_filename = "temperature_log_serialized_v1.2.csv";
// const char * log_filename = "temperature_log_hotspot_mitigation_v1.2.csv";
// const char * log_filename = "temperature_log_hybrid_single_dispatch_v1.2.csv";
// const char * log_filename = "temperature_log_hybrid_double_dispatch_v1.2.csv";
const char * log_filename = "temperature_log_hybrid_single_dispatch_timer_based_v1.0.csv";

std::vector<long> last_total_jiffies(4, 0);
std::vector<long> last_work_jiffies(4, 0);

dispatch_state_t cur_state = CONCURRENT;
timer_t concurrent_timer, serialized_timer;

double read_core_temp(int core_id);
void adjust_thread_mapping(int core_id, int adjustment);
void log_temperatures(const char *log_file);
void initialize_matrices(matrix_data_t *data);
void* multiply_matrices(void * worker);
void set_thread_affinity(pthread_t thread, int core_id);
void serialize_threads_double(int coolest_core1, int coolest_core2);
void deserialize_threads ();
void serialize_threads (int coolest_core);
void switch_to_serialized(int signum);
void switch_to_concurrent(int signum);
void handle_timer_expiration(int signum);
double get_hottest_temp();
double get_coolest_temp();
int find_hottest_core();
int find_coolest_core();
int find_second_coolest_core(int coolest_core);
void set_cpu_freq(int core_id, int freq); 
unsigned long get_cpu_freq(int core_id);
double get_cpu_utilization(int core_id); 
void get_cpu_info();
void get_per_core_utilization(); 
const char* get_core_type(int core_id); 
void print_stat();
void write_sysfs(const char *path, int value); 
void handle_signal(int signal); 
void* log_temp_for_UCB(void* arg); 


double Kp = 1.0, Ki = 0.3, Kd = 0.2;
double integral[NUM_CORES] = {0};
double previous_error[NUM_CORES] = {0};

// Function to initialize configurations
void initialize_configurations() {
    int index = 0;
    for (int Tmin = 55; Tmin <= 62; Tmin++) {
        for (int Tmax = 70; Tmax <= 85; Tmax++) {
            for (int ds = 1; ds <= 10; ds++) {
                for (int dc = 1; dc <= 10; dc++) {
                    for (int fs = 200; fs <= 2000; fs += 100) {
                        for (int fc = 200; fc <= 2000; fc += 100) {
                            configurations[index].Tmin = Tmin;
                            configurations[index].Tmax = Tmax;
                            configurations[index].ds = ds;
                            configurations[index].dc = dc;
                            configurations[index].fs = (fs*1000);
                            configurations[index].fc = (fc*1000);
                            rewards[index] = 0.0;
                            counts[index] = 0;
                            index++;
                        }
                    }
                }
            }
        }
    }
}

// Function to select the configuration using UCB, N = total trials
int select_configuration(int N) {
    double ucb_value, max_ucb = -1;
    int best_config = 0;

    for (int i = 0; i < NUM_CONFIGURATIONS; i++) {
        if (counts[i] == 0) {
            // If the configuration has never been tried, select it
            return i;
        } else {
            // Calculate UCB
            ucb_value = (rewards[i] / counts[i]) + sqrt( (2.0 * log(N)) / counts[i]);
            if (ucb_value > max_ucb) {
                max_ucb = ucb_value;
                best_config = i;
            }
        }
    }
    return best_config;
}

// Function to apply a configuration (adjust DVFS, timer, etc.)
void apply_configuration(Config config) {
    printf("Applying configuration: Tmin=%d, Tmax=%d, ds=%d, dc=%d, fs=%d, fc=%d\n",
           config.Tmin, config.Tmax, config.ds, config.dc, config.fs, config.fc);
    
    MAX_PEAK_TEMP = config.Tmax;
    MIN_AVG_TEMP = config.Tmin;
    BIG_CORE_MIN_FREQ = config.fc;    
    BIG_CORE_MAX_FREQ = config.fs;   
    serialized_duration = config.ds;       
    concurrent_duration = config.dc;   
    return;
}

// Function to calculate the reward
double calculate_reward(double current_temp, double latency) {
    double temp_min = 48.0;
    double temp_max = 90.0;
    double latency_max = 1700.0;
    
    double temp_reward = 1.0 - ((current_temp - temp_min) / (temp_max - temp_min));
    double latency_reward = 1.0 - (latency / latency_max);
    
    double weight_temp = 0.5;
    double weight_latency = 0.5;
    
    double final_reward = (weight_temp * temp_reward) + (weight_latency * latency_reward);
    
    return final_reward;
}


void *UCB (void *args) {
    double current_temp[NUM_CORES];
    double avg_temp = 0.0;
    
    // Start in concurrent mode and initialize the timer for the first switch
    cur_state = CONCURRENT;
    start_timer(&concurrent_timer, concurrent_duration, handle_timer_expiration);  // Start the first timer for 10 seconds

    while (!stop_logging) {
        avg_temp = 0.0;
        for (int i = 0; i < NUM_CORES; i++) {
            current_temp[i] = read_core_temp(i);
            avg_temp += current_temp[i];
        }
        avg_temp /= NUM_CORES;

        printf("Current state: %d, Avg temp: %.2lf\n", cur_state, avg_temp);

        switch (cur_state) {
            case CONCURRENT:
                // Normal operation in concurrent mode, nothing special here
                break;

            case SERIALIZED:
                // Normal operation in serialized mode, nothing special here
                break;

            case WAIT_FOR_THRESHOLD:
                // Timer expired, waiting for temperature condition
                if (avg_temp >= MAX_PEAK_TEMP) {
                    printf("Switching to serialized mode, temperature allows.\n");
                    switch_to_serialized(0);
                } else if (avg_temp <= MIN_AVG_TEMP) {
                    printf("Switching to concurrent mode, temperature allows.\n");
                    switch_to_concurrent(0);
                }
                break;

            default:
                break;
        }

        usleep(MONITOR_INTERVAL);  // Sleep for 5 seconds, adjust as needed
    }
    pthread_exit(NULL);
}


// Signal handler to switch to waiting mode after timer expires
void handle_timer_expiration(int signum) {
    if (cur_state == CONCURRENT) {
        printf("Timer expired, waiting to switch to serialized mode.\n");
        cur_state = WAIT_FOR_THRESHOLD;  // Enter waiting state
    } else if (cur_state == SERIALIZED) {
        printf("Timer expired, waiting to switch to concurrent mode.\n");
        cur_state = WAIT_FOR_THRESHOLD;  // Enter waiting state
    }
}

void start_timer(timer_t *timer_id, int interval_sec, void (*handler)(int)) {
    struct sigaction sa;
    struct itimerspec timer_spec;
    struct sigevent sev;

    // Set up the signal handler
    sa.sa_flags = SA_SIGINFO;
    sa.sa_handler = handler;
    sigemptyset(&sa.sa_mask);

    // Assign a signal number (SIGUSR1 for concurrent, SIGUSR2 for serialized)
    if (timer_id == &concurrent_timer) {
        sigaction(SIGUSR1, &sa, NULL);
    } else {
        sigaction(SIGUSR2, &sa, NULL);
    }

    // Set up the timer event to send signal upon expiration
    sev.sigev_notify = SIGEV_SIGNAL;
    sev.sigev_signo = (timer_id == &concurrent_timer) ? SIGUSR1 : SIGUSR2;
    sev.sigev_value.sival_ptr = timer_id;
    timer_create(CLOCK_REALTIME, &sev, timer_id);

    // Set the timer for the specified interval (seconds)
    timer_spec.it_value.tv_sec = interval_sec;
    timer_spec.it_value.tv_nsec = 0;
    timer_spec.it_interval.tv_sec = 0;  // One-shot timer
    timer_spec.it_interval.tv_nsec = 0;

    timer_settime(*timer_id, 0, &timer_spec, NULL);
}

// Function to get the hottest temperature
double get_hottest_temp() {
    double hottest_temp = read_core_temp(0);  // Initialize with the temperature of the first core
    for (int j = 1; j < NUM_CORES; j++) {
        double temp = read_core_temp(j);  // Get temperature of core j
        if (temp > hottest_temp) {
            hottest_temp = temp;  // Update hottest temperature
        }
    }
    return hottest_temp;
}

// Function to get the coolest temperature
double get_coolest_temp() {
    double coolest_temp = read_core_temp(0);  // Initialize with the temperature of the first core
    for (int j = 1; j < NUM_CORES; j++) {
        double temp = read_core_temp(j);  // Get temperature of core j
        if (temp < coolest_temp) {
            coolest_temp = temp;  // Update coolest temperature
        }
    }
    return coolest_temp;
}


// Function to read the temperature for all cores and find the hottest big core index
int find_hottest_core() {
    double hottest_temp = read_core_temp(0);
    int hottest_core = 0;
    
    for (int i = 1; i < NUM_CORES; i++) {
        double temp = read_core_temp(i);
        if (temp > hottest_temp) {
            hottest_temp = temp;
            hottest_core = i;
        }
    }
    return hottest_core;
}

// Function to find the coolest big core index
int find_coolest_core() {
    double coolest_temp = read_core_temp(0);
    int coolest_core = 0;
    
    for (int i = 1; i < NUM_CORES; i++) {
        double temp = read_core_temp(i);
        if (temp < coolest_temp) {
            coolest_temp = temp;
            coolest_core = i;
        }
    }
    return coolest_core;
}

// Function to find the second coolest core index
int find_second_coolest_core(int coolest_core) {
    int second_coolest_core = (coolest_core == 0) ? 1 : 0;
    double second_coolest_temp = read_core_temp(second_coolest_core);

    for (int i = 0; i < NUM_CORES; i++) {
        if (i != coolest_core) {
            double temp = read_core_temp(i);
            if (temp < second_coolest_temp) {
                second_coolest_temp = temp;
                second_coolest_core = i;
            }
        }
    }
    return second_coolest_core;
}

// Function to set CPU frequency for a given core
void set_cpu_freq(int core_id, int freq) {
    char command[256];
    snprintf(command, sizeof(command), "echo %d | sudo tee /sys/devices/system/cpu/cpu%d/cpufreq/scaling_max_freq > /dev/null", freq, core_id);
    int result = system(command);  // Execute the command to change frequency
    if (result != 0) {
        printf("Failed to set CPU %d frequency to %d KHz\n", core_id, freq);
        return;
    }
    // Get current frequency for this core
    int current_freq = get_cpu_freq(core_id);
    if (current_freq == -1) {
        printf("Unable to verify frequency change for CPU %d\n", core_id);
    } else if (current_freq != freq) {
        printf("Failed to change CPU %d frequency to %d KHz, current frequency is %d KHz\n", core_id, freq, current_freq);
    } else {
        printf("Successfully set CPU %d frequency to %d KHz\n", core_id, freq);
    }
        
}


// Signal handler to switch to serialized mode
void switch_to_serialized(int signum) {
    printf("Switching to serialized mode due to timer event.\n");
    set_cpu_freq(find_hottest_core()+4, BIG_CORE_MAX_FREQ);
    serialize_threads (find_coolest_core()+4);  // Example core IDs, adjust as needed
    cur_state = SERIALIZED;
    // Start a new timer to switch back to concurrent mode after serialized duration
    start_timer(&serialized_timer, serialized_duration, handle_timer_expiration);  // 5 seconds in serialized mode
}

// Signal handler to switch to concurrent mode
void switch_to_concurrent(int signum) {
    printf("Switching to concurrent mode due to timer event.\n");
    set_cpu_freq(find_hottest_core()+4, BIG_CORE_MIN_FREQ);
    deserialize_threads();
    cur_state = CONCURRENT;
    // Start a new timer to switch back to serialized mode after concurrent duration
    start_timer(&concurrent_timer, concurrent_duration, handle_timer_expiration);  // 10 seconds in concurrent mode
}


// Function to get the current frequency of a core
unsigned long get_cpu_freq(int core_id) {
    char path[100];
    unsigned long freq = 0;
    FILE *fp;

    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", core_id);
    
    fp = fopen(path, "r");
    if (fp == NULL) {
        perror("Failed to open CPU frequency file");
        return 0;
    }

    if (fscanf(fp, "%lu", &freq) != 1) {
        perror("Failed to read frequency");
        fclose(fp);
        return 0;
    }

    fclose(fp);
    return freq;
}


// Function to read CPU utilization for a specific core (core_id)
double get_cpu_utilization(int core_id) {
    FILE *fp;
    char buffer[1024];
    char cpu_label[10];
    unsigned long user, nice, system, idle, iowait, irq, softirq, steal;
    unsigned long total_jiffies, work_jiffies;

    snprintf(cpu_label, sizeof(cpu_label), "cpu%d", core_id);
    
    fp = fopen("/proc/stat", "r");
    if (fp == NULL) {
        perror("Failed to open /proc/stat");
        return 0;
    }

    while (fgets(buffer, sizeof(buffer), fp)) {
        if (strstr(buffer, cpu_label)) {
            sscanf(buffer, "%*s %lu %lu %lu %lu %lu %lu %lu %lu",
                   &user, &nice, &system, &idle, &iowait, &irq, &softirq, &steal);
            work_jiffies = user + nice + system;
            total_jiffies = work_jiffies + idle + iowait + irq + softirq + steal;
            fclose(fp);
            return (double)work_jiffies / total_jiffies * 100.0; // Return percentage
        }
    }

    fclose(fp);
    return 0;
}


void get_cpu_info() {
    FILE *fp;
    char output[1035];

    // Run the lscpu command
    fp = popen("lscpu", "r");
    if (fp == NULL) {
        printf("Failed to run command\n");
        exit(1);
    }

    // Read the output a line at a time
    while (fgets(output, sizeof(output) - 1, fp) != NULL) {
        printf("%s", output);
    }

    // Close the command
    pclose(fp);
}

void get_cpu_utilization() {
    FILE *fp;
    char output[1035];

    // Run the top command to get CPU usage
    fp = popen("top -bn1 | grep 'Cpu(s)'", "r");
    if (fp == NULL) {
        printf("Failed to run command\n");
        exit(1);
    }

    // Read the output a line at a time
    while (fgets(output, sizeof(output) - 1, fp) != NULL) {
        printf("%s", output);
    }

    // Close the command
    pclose(fp);
}

void get_per_core_utilization() {
    FILE *fp;
    char output[1035];

    // Run the mpstat command to get CPU usage per core
    fp = popen("mpstat -P ALL 1 1", "r");
    if (fp == NULL) {
        printf("Failed to run command\n");
        exit(1);
    }

    // Read the output a line at a time
    while (fgets(output, sizeof(output) - 1, fp) != NULL) {
        printf("%s", output);
    }

    // Close the command
    pclose(fp);
}

const char* get_core_type(int core_id) {
    if (core_id >= 0 && core_id <= 3) {
        return "LITTLE";
    } else if (core_id >= 4 && core_id <= 7) {
        return "BIG";
    }
    return "UNKNOWN";
}


void print_stat()
{
    int num_cores = 8;  // XU4 has 8 cores (4 big, 4 little)
    unsigned long freq;
    double utilization;
    
    for (int core_id = 0; core_id < 8; core_id++) {
        // Get core type
        const char *core_type = get_core_type(core_id);

        // Get current frequency for this core
        freq = get_cpu_freq(core_id);
        
        // Get current utilization for this core
        utilization = get_cpu_utilization(core_id);

        // Print frequency and utilization
        printf("Core %d (%s): Frequency = %lu kHz, Utilization = %.2f%%\n",
               core_id, core_type, freq, utilization);
    }

    // get_cpu_info();
    // get_cpu_utilization();
    // get_per_core_utilization();
    
    return;
}

void *hybrid_single_dispatch_timer_based(void *args) {
    double current_temp[NUM_CORES];
    double avg_temp = 0.0;

    // Start in concurrent mode and initialize the timer for the first switch
    cur_state = CONCURRENT;
    start_timer(&concurrent_timer, concurrent_duration, handle_timer_expiration);  // Start the first timer for 10 seconds

    while (!stop_logging) {
        avg_temp = 0.0;
        for (int i = 0; i < NUM_CORES; i++) {
            current_temp[i] = read_core_temp(i);
            avg_temp += current_temp[i];
        }
        avg_temp /= NUM_CORES;

        printf("Current state: %d, Avg temp: %.2lf\n", cur_state, avg_temp);

        printf("before print\n");
        print_stat();
        printf("after print\n");

        switch (cur_state) {
            case CONCURRENT:
                // Normal operation in concurrent mode, nothing special here
                break;

            case SERIALIZED:
                // Normal operation in serialized mode, nothing special here
                break;

            case WAIT_FOR_THRESHOLD:
                // Timer expired, waiting for temperature condition
                if (avg_temp >= MAX_PEAK_TEMP) {
                    printf("Switching to serialized mode, temperature allows.\n");
                    switch_to_serialized(0);
                } else if (avg_temp <= MIN_AVG_TEMP) {
                    printf("Switching to concurrent mode, temperature allows.\n");
                    switch_to_concurrent(0);
                }
                break;

            default:
                break;
        }

        usleep(MONITOR_INTERVAL);  // Sleep for 5 seconds, adjust as needed
    }
    pthread_exit(NULL);
}


void * DVFS_based_thermal(void * args) {
    double current_temp[NUM_CORES];
    double avg_temp = 0.0;
    dispatch_state_t current_state = CONCURRENT;
    
    while (!stop_logging)
    {
        avg_temp = 0.0;
        for (int i = 0; i < NUM_CORES; i++) {
            current_temp[i] = read_core_temp(i);
            avg_temp += current_temp[i];
        }
        avg_temp /= NUM_CORES;
        printf("current state: %d avg temp %.2lf\n", current_state, avg_temp);

        int hottest_core = 0;
        int coolest_core = 0;
        double hottest_temp = current_temp[0];
        double coolest_temp = current_temp[0];
        
        // Find the hottest core
        for (int j = 1; j < NUM_CORES; j++) {
            if (current_temp[j] > hottest_temp) {
                hottest_core = j;
                hottest_temp = current_temp[hottest_core];
            }
        }

        // Find the coolest core
        for (int j = 1; j < NUM_CORES; j++) {
            if (current_temp[j] < coolest_temp) {
                coolest_core = j;
                coolest_temp = current_temp[coolest_core];
            }
        }

        printf("hottest core %d, temp %.2lf\n", hottest_core+4, hottest_temp);
        printf("coolest core %d, temp %.2lf \n", coolest_core+4, coolest_temp);

        switch (current_state)
        {
        case CONCURRENT:
            if(avg_temp >= MAX_PEAK_TEMP) {
                // Reduce the frequency of the hottest core to lower temperature
                set_cpu_freq(hottest_core+4, BIG_CORE_MIN_FREQ);
                current_state = REDUCED_FREQ;
            }
            break;

        case REDUCED_FREQ:
            if(avg_temp <= MIN_AVG_TEMP) {
                // Restore the frequency of the hottest core to its normal value
                set_cpu_freq(hottest_core+4, BIG_CORE_MAX_FREQ);
                current_state = CONCURRENT;
            }
            break;

        default:
            break;
        }

        usleep(MONITOR_INTERVAL);
    }
    pthread_exit(NULL);
}

void write_sysfs(const char *path, int value) {
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        perror("Failed to open sysfs file");
        exit(EXIT_FAILURE);
    }
    fprintf(fp, "%d", value);
    fclose(fp);
}

// Signal handler to catch Ctrl+C (SIGINT)
void handle_signal(int signal) {
    if (signal == SIGINT) {
        printf("\nCaught signal %d (SIGINT). Turning automatic mode back on.\n", signal);
        stop_logging = 1;
        write_sysfs(automatic_path, 1);  // Turn automatic mode back on
        exit(EXIT_SUCCESS);
    }
}

void * concurrent (void * args) {
    pthread_exit(NULL);
}

void * serialized (void * args) {
    int big_core_id = *(int *)args;
    for(int i=0; i<NUM_THREADS; i++){
        set_thread_affinity(thread_args[i].thread, big_core_id);
        thread_args[i].core_id = big_core_id;
    }
    return NULL;
}

void * hotspot_mitigation (void * args) {
    double current_temp[NUM_CORES];
    while (!stop_logging)
    {
        for (int i = 0; i < NUM_CORES; i++) {
            current_temp[i] = read_core_temp(i);
        }

        int hottest_core = 0;
        int coolest_core = 0;
        double hottest_temp = current_temp[0];
        double coolest_temp = current_temp[0];
        for (int j = 1; j < NUM_CORES; j++) {
            if (current_temp[j] > current_temp[hottest_core]) {
                hottest_core = j;
                hottest_temp = current_temp[hottest_core];
            }
        }
        
        printf("hottest core %d, temp %.2lf\n", hottest_core+4, hottest_temp);
        
        for (int j = 1; j < NUM_CORES; j++) {
            if (current_temp[j] < current_temp[coolest_core]) {
                coolest_core = j;
                coolest_temp = current_temp[coolest_core];
            }
        }
        printf("coolest core %d, temp %.2lf \n", coolest_core+4, coolest_temp);

        hottest_core += 4;
        coolest_core += 4;
        
        for (int j = 0; j < NUM_THREADS; j++) {
            int diff = thread_args[j].core_id - hottest_core;
            if (diff == 0) {
                set_thread_affinity(thread_args[j].thread, coolest_core);
                thread_args[j].core_id = coolest_core;
                printf(" thread id %d, new core %d\n", thread_args[j].thread_id, thread_args[j].core_id);
                break;
            }
        }
        
        usleep(MONITOR_INTERVAL);
    }
    pthread_exit(NULL);
}

void serialize_threads (int coolest_core) {
    for (int j = 0; j < NUM_THREADS; j++) {
        set_thread_affinity(thread_args[j].thread, coolest_core);
        thread_args[j].core_id = coolest_core;
        printf("thread id %d, new core %d\n", thread_args[j].thread_id, thread_args[j].core_id);
    }
}


void deserialize_threads () {
    for (int j = 0; j < NUM_THREADS; j++) {
        set_thread_affinity(thread_args[j].thread, j+4);
        thread_args[j].core_id = j+4;
        printf("thread id %d, new core %d\n", thread_args[j].thread_id, thread_args[j].core_id);
    }
}

void * hybrid_single_dispatch (void * args) {
    double current_temp[NUM_CORES];
    double avg_temp = 0.0;
    dispatch_state_t current_state = CONCURRENT;
    while (!stop_logging)
    {
        avg_temp = 0.0;
        for (int i = 0; i < NUM_CORES; i++) {
            current_temp[i] = read_core_temp(i);
            avg_temp += current_temp[i];
        }
        avg_temp /= NUM_CORES;
        printf("current state: %d avg temp %.2lf\n", current_state, avg_temp);

        int hottest_core = 0;
        int coolest_core = 0;
        double hottest_temp = current_temp[0];
        double coolest_temp = current_temp[0];
        for (int j = 1; j < NUM_CORES; j++) {
            if (current_temp[j] > current_temp[hottest_core]) {
                hottest_core = j;
                hottest_temp = current_temp[hottest_core];
            }
        }
        
        printf("hottest core %d, temp %.2lf\n", hottest_core+4, hottest_temp);
        
        for (int j = 1; j < NUM_CORES; j++) {
            if (current_temp[j] < current_temp[coolest_core]) {
                coolest_core = j;
                coolest_temp = current_temp[coolest_core];
            }
        }
        printf("coolest core %d, temp %.2lf \n", coolest_core+4, coolest_temp);

        hottest_core += 4;
        coolest_core += 4;
        
        switch (current_state)
        {
        case CONCURRENT:
            if(avg_temp >= MAX_PEAK_TEMP) {
                printf("switching to serialized mode in single dispatch\n");
                serialize_threads(coolest_core);
                current_state = SERIALIZED;
            }
            break;

        case SERIALIZED:
            if(avg_temp <= MIN_AVG_TEMP) {
                printf("switching to concurrent mode in single dispatch\n");
                deserialize_threads();
                current_state = CONCURRENT;
            }
            break;

        default:
            break;
        }

        usleep(MONITOR_INTERVAL);
    }
    pthread_exit(NULL);
}

void serialize_threads_double(int coolest_core1, int coolest_core2) {
    printf("Serializing threads across cores %d and %d\n", coolest_core1, coolest_core2);
    for (int i = 0; i < NUM_THREADS; i++) {
        int target_core = (i % 2 == 0) ? coolest_core1 : coolest_core2;  // Alternate between the two cores
        set_thread_affinity(thread_args[i].thread, target_core);
        thread_args[i].core_id = target_core;
        printf("thread id %d, new core %d\n", thread_args[i].thread_id, thread_args[i].core_id);
    }
}

void * hybrid_double_dispatch (void * args) {
    double current_temp[NUM_CORES];
    double avg_temp = 0.0;
    dispatch_state_t current_state = CONCURRENT;
    while (!stop_logging)
    {
        avg_temp = 0.0;
        for (int i = 0; i < NUM_CORES; i++) {
            current_temp[i] = read_core_temp(i);
            avg_temp += current_temp[i];
        }
        avg_temp /= NUM_CORES;
        printf("current state: %d avg temp %.2lf\n", current_state, avg_temp);

        int hottest_core = 0;  

        int coolest_core1 = 0;
        int coolest_core2 = 1;  

        double hottest_temp = current_temp[0];

        double coolest_temp1 = current_temp[0];
        double coolest_temp2 = current_temp[1];

        for (int j = 1; j < NUM_CORES; j++) {
            if (current_temp[j] > current_temp[hottest_core]) {
                hottest_core = j;
                hottest_temp = current_temp[hottest_core];
            }
        }
        
        printf("hottest core %d, temp %.2lf\n", hottest_core+4, hottest_temp);
        
        // Find the two coolest cores
        for (int j = 1; j < NUM_CORES; j++) {
            if (current_temp[j] < coolest_temp1) {
                coolest_temp2 = coolest_temp1;  // The previous coolest becomes the second coolest
                coolest_core2 = coolest_core1;
                
                coolest_temp1 = current_temp[j];  // New coolest
                coolest_core1 = j;
            } else if (current_temp[j] < coolest_temp2) {
                coolest_temp2 = current_temp[j];
                coolest_core2 = j;
            }
        }

        printf("coolest cores %d and %d, temps %.2lf and %.2lf\n", 
               coolest_core1 + 4, coolest_core2 + 4, coolest_temp1, coolest_temp2);

        
        hottest_core += 4;
        coolest_core1 += 4;
        coolest_core2 += 4;
        
        switch (current_state)
        {
        case CONCURRENT:
            if(avg_temp >= MAX_PEAK_TEMP) {
                printf("switching to serialized mode in double dispatch\n");
                serialize_threads_double(coolest_core1, coolest_core2);
                current_state = SERIALIZED;
            }
            break;

        case SERIALIZED:
            if(avg_temp <= MIN_AVG_TEMP) {
                printf("switching to concurrent mode in single dispatch\n");
                deserialize_threads();
                current_state = CONCURRENT;
            }
            break;

        default:
            break;
        }

        usleep(MONITOR_INTERVAL);
    }
    pthread_exit(NULL);
}

void parse_cpu_line(const std::string& line, int core_id, long& total_jiffies, long& work_jiffies) {
    std::istringstream iss(line);
    std::string cpu;
    long user, nice, system, idle, iowait, irq, softirq, steal;

    iss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;

    if (cpu == "cpu" + std::to_string(core_id)) {
        work_jiffies = user + nice + system;
        total_jiffies = work_jiffies + idle + iowait + irq + softirq + steal;
    }
}

double read_core_load(int core_id) {
    std::ifstream stat_file("/proc/stat");
    std::string line;
    long total_jiffies = 0, work_jiffies = 0;

    while (std::getline(stat_file, line)) {
        parse_cpu_line(line, core_id, total_jiffies, work_jiffies);
    }

    // Calculate the CPU load as a percentage
    long total_diff = total_jiffies - last_total_jiffies[core_id];
    long work_diff = work_jiffies - last_work_jiffies[core_id];

    last_total_jiffies[core_id] = total_jiffies;
    last_work_jiffies[core_id] = work_jiffies;

    if (total_diff == 0) return 0.0;  // Avoid division by zero

    double load = (100.0 * work_diff) / total_diff;
    return load;
}

// Comparator for sorting cores by weighted temperature
bool compare_cores(const std::pair<int, double>& a, const std::pair<int, double>& b) {
    return a.second < b.second;
}

void * pid_control(void * args) {
    double current_temp[NUM_CORES];
    double error[NUM_CORES];
    double output[NUM_CORES];

    double current_load[NUM_CORES];

    while (!stop_logging) {
        for (int i = 0; i < NUM_CORES; i++) {
            current_temp[i] = read_core_temp(i);
            error[i] = TARGET_TEMP - current_temp[i];

            current_load[i] = read_core_load(i);
            
            integral[i] += error[i];
            double derivative = error[i] - previous_error[i];
            
            output[i] = Kp * error[i] + Ki * integral[i] + Kd * derivative;
            
            previous_error[i] = error[i];
        }
        
        for (int i = 0; i < NUM_CORES; i++) {
            printf("PID control output for core %d: %.2lf\n", i + 4, output[i]);
            printf("core : %d, current temp: %.2lf current cpu load: %.2lf\n", i+4, current_temp[i], current_load[i]);
        }

        // Step 2: Calculate weighted temperature for each core (0.7 * temp + 0.3 * load)
        std::vector<std::pair<int, double>> weighted_temps(NUM_CORES);
        for (int i = 0; i < NUM_CORES; i++) {
            weighted_temps[i] = {i, 0.7 * current_temp[i] + 0.3 * current_load[i]};
            printf("weighted temp: first = %d, second = %.2lf\n", weighted_temps[i].first, weighted_temps[i].second);
        }

        // Step 3: Sort cores by weighted temperature (ascending order)
        std::sort(weighted_temps.begin(), weighted_temps.end(), compare_cores);
        for (int i = 0; i < NUM_CORES; i++) {
            printf("weighted temp: first = %d, second = %.2lf\n", weighted_temps[i].first, weighted_temps[i].second);
        }

        // Step 4: Redistribute workload based on PID output and weighted temperatures
        int coolest_index = 0;
        int hottest_index = NUM_CORES - 1;

        while (hottest_index > coolest_index) {
            int hottest_core = weighted_temps[hottest_index].first;
            int coolest_core = weighted_temps[coolest_index].first;

            printf("in while: hottest core %d, coolest core %d\n", hottest_core, coolest_core);

            double hottest_core_load = output[hottest_core];
            double coolest_core_capacity = -output[coolest_core];

            printf("hottest_core_load = %.2lf, coolest_core_capacity = %.2lf\n", hottest_core_load, coolest_core_capacity);

            if (hottest_core_load > 0 &&  coolest_core_capacity < hottest_core_load) {
                double offload_amount = std::min(hottest_core_load, coolest_core_capacity);

                std::cout << "Offloading " << offload_amount << " load from core " << hottest_core + 4 
                          << " to core " << coolest_core + 4 << std::endl;

                // Migrate a thread from hottest_core to coolest_core
                for (auto& thread_arg : thread_args) {
                    if (thread_arg.core_id == hottest_core + 4) {
                        set_thread_affinity(thread_arg.thread, coolest_core + 4);
                        thread_arg.core_id = coolest_core + 4;
                        break;
                    }
                }

                // Update PID outputs
                output[hottest_core] -= offload_amount;
                output[coolest_core] += offload_amount;

                // Re-evaluate the weighted temperatures after redistribution
                weighted_temps[hottest_index].second -= offload_amount;
                weighted_temps[coolest_index].second += offload_amount;

                if (output[hottest_core] <= 0) {
                    hottest_index--;
                }
                if (output[coolest_core] >= 0) {
                    coolest_index++;
                }
            } else {
                if (hottest_core_load <= 0) hottest_index--;
                if (coolest_core_capacity <= 0) coolest_index++;
            }
        }
        
        usleep(MONITOR_INTERVAL);
    }
    pthread_exit(NULL);
}

double read_core_temp(int core_id) {
    FILE *fp = fopen(big_core_paths[core_id], "r");
    if (fp == NULL) {
        perror("Failed to open temperature sensor file");
        return -1;
    }

    double temp;
    fscanf(fp, "%lf", &temp);
    fclose(fp);

    return temp / 1000.0;
}

void adjust_thread_mapping(int core_id, int adjustment) {
    printf("Adjusting core %d by %d threads\n", core_id, adjustment);
}

void log_temperatures(const char *log_file) {
    FILE *fp = fopen(log_file, "w");
    if (fp == NULL) {
        perror("Failed to open log file");
        return;
    }

    fprintf(fp, "Time,Big Core 0,Big Core 1,Big Core 2,Big Core 3\n");
    int t=0;
    while(!stop_logging)
    {
        for (int n = 0; n < LOG_DURATION; n++) {
            fprintf(fp, "%d", t++);
            for (int i = 0; i < NUM_CORES; i++) {
                fprintf(fp, ",%.2f", read_core_temp(i));
            }
            fprintf(fp, "\n");
            fflush(fp);
        }
        usleep(LOG_INTERVAL);
    }
    fclose(fp);
    pthread_exit(NULL);
}

// Temperature Logging Thread with cancellation enabled, Return Per big core avg temp
void* log_temp_for_UCB(void* arg) {
    double cumulative_temp_sum[NUM_CORES];
    memset(cumulative_temp_sum, 0.0, sizeof(cumulative_temp_sum));  // Initialize cumulative temp sum
    int N=0;

    // Enable cancellation
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
    pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, NULL);

    while (1) {
        pthread_testcancel();  // Allow cancellation at this point

        //pthread_mutex_lock(&temp_mutex);
        for (int i = 0; i < NUM_CORES; i++) {
            cumulative_temp_sum[i] += read_core_temp(i);  // Accumulate temperature for this core
        }
        //pthread_mutex_unlock(&temp_mutex);
        ++N;

        usleep(50000);  // Sleep for 50 ms
    }

    for (int i = 0; i < NUM_CORES; i++) {
        cumulative_temp_sum[i] = (cumulative_temp_sum[i]/N);  // Accumulate temperature for this core
    }

    pthread_exit(cumulative_temp_sum);  // Return the cumulative temperature sum on exit
}

void initialize_matrices(matrix_data_t *data) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            data->A[i][j] = rand() % 100;
            data->B[i][j] = rand() % 100;
            data->C[i][j] = 0.0;
        }
    }
}

void* multiply_matrices(void * worker) {
    thread_arg_t * args = (thread_arg_t *) worker;
    int thread_id = args->thread_id;
    matrix_data_t *data = args->data;

    pthread_t temp_logger;
    void* temp_logger_result;

    struct timeval start, end;
    double latency;
    
    char filename[50];
    sprintf(filename, "thread_%d_execution_times.csv", thread_id);

    FILE *log_file = fopen(filename, "w");
    if (log_file == NULL) {
        perror("Error opening file");
        pthread_exit(NULL);
    }
    
    fprintf(log_file, "ExecutionTime(ms)\n");

    initialize_configurations();
    total_trials = 0;
    selected_config = select_configuration(total_trials);
    apply_configuration(configurations[selected_config]);

    while(!stop_logging){

        // Wait for all threads to reach this point
        pthread_barrier_wait(&barrier);

        if (thread_id == 0) {
            // Start the temperature logging thread
            pthread_create(&temp_logger, NULL, log_temp_for_UCB, NULL);
        }

        // Wait for all threads to reach this point
        pthread_barrier_wait(&barrier);
        
        gettimeofday(&start, NULL);

        printf("thread id %d, core id using getcpu %d, core id %d \n", thread_id, sched_getcpu(), args->core_id);
        for (int i = 0; i < MATRIX_SIZE; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                for (int k = 0; k < MATRIX_SIZE; k++) {
                    data->C[i][j] += data->A[i][k] * data->B[k][j];
                }
            }
        }

        gettimeofday(&end, NULL);
        
        latency = (end.tv_sec - start.tv_sec) * 1000.0;  // sec to ms
        latency += (end.tv_usec - start.tv_usec) / 1000.0;  // us to ms

        // Store the latency in a shared array
        execution_times[thread_id] = latency;

        fprintf(log_file, "%.2lf\n", latency);
        fflush(log_file);

        // Wait for all threads to reach this point
        pthread_barrier_wait(&barrier);

        // After the barrier, thread 0 will compute the average execution time
        if (thread_id == 0) {
            // Cancel the temperature logging thread
            pthread_cancel(temp_logger);

            // Join the temperature logging thread and collect the result
            pthread_join(temp_logger, &temp_logger_result);

            // Print the cumulative temperature sum for this round
            double* per_big_core_avg_temp = (double*)temp_logger_result;
            double avg_temp = 0.0;
            for (int i = 0; i < NUM_CORES; i++) {
                printf("Core %d: %.2lf°C\n", i + 4, per_big_core_avg_temp[i]);
                avg_temp += per_big_core_avg_temp[i];
            }
            avg_temp /= NUM_CORES;

            double total_latency = 0.0;
            for (int i = 0; i < NUM_THREADS; i++) {
                total_latency += execution_times[i];
            }
            double avg_latency = (total_latency / NUM_THREADS);
            printf("Average Execution Time: %.2lf ms\n", avg_latency);

            // Here, you can update the reward function or log avg_latency for UCB learning
            // Example: Update UCB, change DVFS, etc.
            
            // latency = measure_latency();
            // temp = measure_avg_temp();

            double reward = calculate_reward(avg_latency, avg_temp);
            rewards[selected_config] += reward;
            ++counts[selected_config];

            ++total_trials;
            selected_config = select_configuration(total_trials);
            apply_configuration(configurations[selected_config]);
        }

        // Wait again to synchronize before starting next multiplication
        pthread_barrier_wait(&barrier);

        usleep((0.009*1000000)); 
    }

    fclose(log_file);
    pthread_exit(NULL);
}

void set_thread_affinity(pthread_t thread, int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
}

void* monitor_temperatures(void* arg) {
    log_temperatures(log_filename);
    pthread_exit(NULL);
}

void merge_logs() {
    // System call to run the Python script
    system("/bin/python3 merge_logs.py");
}

int main() {
    // Register the merge_logs function to run at exit
    atexit(merge_logs);

    signal(SIGINT, handle_signal);
    
    printf("Starting experiment. Press Ctrl+C to exit and re-enable automatic fan control.\n");
    
    pthread_mutex_init(&lock, NULL);
    // Initialize the barrier
    pthread_barrier_init(&barrier, NULL, NUM_THREADS);


    int i;
    for (i = 0; i < NUM_THREADS; i++) {
        initialize_matrices(&matrices[i]);
    }
    
    for (i = 0; i < NUM_THREADS; i++) {
        thread_args[i].thread_id = i;
        thread_args[i].data = &matrices[i];
        thread_args[i].core_id = i + 4; // Initial core assignment to big cores 4, 5, 6, 7
        pthread_create(&threads[i], NULL, multiply_matrices, (void *)&thread_args[i]);
        thread_args[i].thread = threads[i];
        set_thread_affinity(thread_args[i].thread, thread_args[i].core_id);
    }

    log_filename = "temperature_log_UCB_v1.0.csv";
    pthread_create(&temp_thread, NULL, monitor_temperatures, NULL);
    set_thread_affinity(temp_thread, 0);
    
    int big_core_id = 4;
    // pthread_create(&pid_thread, NULL, concurrent, NULL);
    // pthread_create(&pid_thread, NULL, serialized, &big_core_id);
    // pthread_create(&pid_thread, NULL, hotspot_mitigation, NULL);
    // pthread_create(&pid_thread, NULL, hybrid_single_dispatch, NULL);
    // pthread_create(&pid_thread, NULL, hybrid_double_dispatch, NULL);
    // pthread_create(&pid_thread, NULL, hybrid_single_dispatch_timer_based, NULL);
    pthread_create(&pid_thread, NULL, UCB, NULL);

    set_thread_affinity(pid_thread, 1);
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    pthread_join(temp_thread, NULL);
    pthread_join(pid_thread, NULL);

    // Destroy the barrier after use
    pthread_barrier_destroy(&barrier);

    return 0;
}

