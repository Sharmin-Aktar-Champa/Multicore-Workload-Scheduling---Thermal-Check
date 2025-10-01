#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>

#define NUM_CORES 4
#define TARGET_TEMP 80.0
#define MATRIX_SIZE 256
#define NUM_THREADS 4
#define LOG_INTERVAL 60000
#define LOG_DURATION 1

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

pthread_t threads[NUM_THREADS];
pthread_t temp_thread;
thread_arg_t thread_args[NUM_THREADS];
matrix_data_t matrices[NUM_THREADS];
pthread_t pid_thread;
pthread_mutex_t lock; 

volatile int stop_logging = 0;
const char *big_core_paths[NUM_CORES] = {
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/thermal/thermal_zone1/temp",
        "/sys/class/thermal/thermal_zone2/temp",
        "/sys/class/thermal/thermal_zone3/temp"
    };

double read_core_temp(int core_id);
void adjust_thread_mapping(int core_id, int adjustment);
void log_temperatures(const char *log_file);
void initialize_matrices(matrix_data_t *data);
void* multiply_matrices(void * worker);
void set_thread_affinity(pthread_t thread, int core_id);

double Kp = 1.0, Ki = 0.1, Kd = 0.01;
double integral[NUM_CORES] = {0};
double previous_error[NUM_CORES] = {0};

void * pid_control(void * args) {
    double current_temp[NUM_CORES];
    double error[NUM_CORES];
    double output[NUM_CORES];

    while (1) {
        for (int i = 0; i < NUM_CORES; i++) {
            current_temp[i] = read_core_temp(i);
            error[i] = TARGET_TEMP - current_temp[i];
            
            integral[i] += error[i];
            double derivative = error[i] - previous_error[i];
            
            output[i] = Kp * error[i] + Ki * integral[i] + Kd * derivative;
            
            previous_error[i] = error[i];
        }
        
        for(int i=0; i<NUM_CORES; i++){
            printf("pid control output %.2lf \n", output[i]);
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
        
        usleep(500000);
    }
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
    
    while(1){
        printf("thread id %d, core id using getcpu %d, core id %d \n", thread_id, sched_getcpu(), args->core_id);
        for (int i = 0; i < MATRIX_SIZE; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                for (int k = 0; k < MATRIX_SIZE; k++) {
                    data->C[i][j] += data->A[i][k] * data->B[k][j];
                }
            }
        }

        usleep(100000); 
    }
    pthread_exit(NULL);
}

void set_thread_affinity(pthread_t thread, int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
}

void* monitor_temperatures(void* arg) {
    log_temperatures("temperature_log.csv");
    pthread_exit(NULL);
}

int main() {
    pthread_mutex_init(&lock, NULL);
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
    
    pthread_create(&temp_thread, NULL, monitor_temperatures, NULL);
    set_thread_affinity(temp_thread, 0);
    
    pthread_create(&pid_thread, NULL, pid_control, NULL);
    set_thread_affinity(pid_thread, 1);
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    stop_logging = 1;
    pthread_join(temp_thread, NULL);
    pthread_join(pid_thread, NULL);
    return 0;
}

