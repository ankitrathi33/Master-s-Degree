#include<stdio.h>
#include<pthread.h>
#include<semaphore.h>
#include<stdlib.h>

sem_t mutex, empty, full;
int queue[2], avail;

void *producer[void);
void *consumer[void);

/*Thread T0*/
int main(void) {
	pthread_t prod_h, cons_h;

	avail = 0;
	sem_init(&mutex, 0, 1);
	sem_init(&empty, 0, 1);
	sem_init(&full, 0, 0);
	pthread_create(&prod_h, o, producer, NULL);
	pthread_create(&cons_h, 0, consumer, NULL);
	pthread_join(prod_h, 0);
	pthread_join(cons_h, 0);
	exit(0);
}

/*Thread T1*/
void *producer(void) {
	int prod = 0, item;
	while (prod < 2) {
		item = rand() % 1000;
		sem_wait(&empty);
		sem_wait(&mutex);
		queue[avail] = item;
		avail++; 
		prod++;
		sem_post(&mutex);
		sem_post(&full);

		}
	pthread_exit(0);
}
/*Thread T2*/
void *consumer(void) {
	int cons = 0, my_item;
	while (cons < 2) {
		sem_wait(&full);
		sem_wait(&mutex);
		cons++;
		avail--;
		my_item = queue[avail];
		sem_post(&mutex);
		sem_post(&empty);
		printf("Consumed:%d\n", my_item);

	}
	pthread_exit(0);
}





