// pthread version
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pthread.h>
#include <list>

double image_left;
double image_right;
double image_lower;
double image_upper;
int image_width;
int image_height;
int *image = nullptr;

struct Block
{
    int h0, h1, v0, v1, iters;
    Block(int _h0, int _h1, int _v0, int _v1, int _iters)
    {
        h0 = _h0;
        h1 = _h1;
        v0 = _v0;
        v1 = _v1;
        iters = _iters;
    }
};

struct Task
{
    void (*func)(void *);
    void *args;
};

class Threadpool
{
private:
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    std::list<pthread_t> threadq;
    std::list<Task> taskq;
    int shutdown;

    enum SHUTDOWN_MODE
    {
        WAIT_TASK,
        NORMAL_SHUTDOWN,
        MANDATORY_SHUTDOWN
    };
    static void *excute(void *threadpool);

public:
    Threadpool(size_t n_thread);
    void submit_task(const Task &task) { taskq.push_back(task); }
    void run_task();
};

Threadpool::Threadpool(size_t n_thread)
{
    /* Initialize */
    shutdown = NORMAL_SHUTDOWN;

    /* Allocate thread and task queue */
    for (size_t i = 0; i < n_thread; ++i)
        threadq.push_back(pthread_t());

    /* Initialize mutex and conditional variable first */
    assert(pthread_mutex_init(&mutex, NULL) == 0);
    assert(pthread_cond_init(&cond, NULL) == 0);

#ifdef DEBUG
    printf("Create a threadpool.\n");
#endif // DEBUG
}
void *Threadpool::excute(void *threadpool)
{
    Threadpool *tp = (Threadpool *)threadpool;

    while (true)
    {
/* Lock must be taken to wait on conditional variable */
#ifdef DEBUG
        printf("%d lock.\n", pthread_self());
#endif // DEBUG
        pthread_mutex_lock(&(tp->mutex));

        /* Wait on condition variable, check for spurious wakeups.
           When returning from pthread_cond_wait(), we own the lock. */
        // while (tp->taskq.empty() && tp->shutdown == WAIT_TASK)
        //     pthread_cond_wait(&(tp->cond), &(tp->mutex));

        if ((tp->shutdown == MANDATORY_SHUTDOWN) || ((tp->shutdown == NORMAL_SHUTDOWN) && tp->taskq.empty()))
            break;

        /* Grab next task */
        Task task(tp->taskq.front());
        tp->taskq.pop_front();

/* Unlock */
#ifdef DEBUG
        printf("%d unlock.\n", pthread_self());
#endif // DEBUG
        pthread_mutex_unlock(&(tp->mutex));

        /* Get to work */
        (*(task.func))(task.args);
    }
#ifdef DEBUG
    printf("%d unlock.\n", pthread_self());
#endif // DEBUG
    pthread_mutex_unlock(&(tp->mutex));

    return NULL;
}
void Threadpool::run_task()
{
    /* Start worker threads */
    for (auto &t : threadq)
    {
#ifdef DEBUG
        printf("Thread %lu running.\n", t);
#endif // DEBUG
        assert(pthread_create(&t, NULL, &Threadpool::excute, this) == 0);
    }
    for (auto &t : threadq)
    {
        assert(pthread_join(t, NULL) == 0);
    }
}

void write_png(const char *filename, int iters, int width, int height)
{
    const int *buffer = image;
    FILE *fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y)
    {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x)
        {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters)
            {
                if (p & 16)
                {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                }
                else
                {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void render(void *args)
{
    Block *blk = (Block *)args;
    const int img_height = image_height;
    const int img_width = image_width;
    const double img_left = image_left;
    const double img_right = image_right;
    const double img_lower = image_lower;
    const double img_upper = image_upper;

#ifdef DEBUG
    printf("Rendering %d, %d, %d, %d\n", blk->h0, blk->h1, blk->v0, blk->v1);
#endif // DEBUG

    for (int j = blk->v0; j < blk->v1; ++j)
    {
        double y0 = j * ((img_upper - img_lower) / img_height) + img_lower;
        for (int i = blk->h0; i < blk->h1; ++i)
        {
            double x0 = i * ((img_right - img_left) / img_width) + img_left;

            int repeats = 0;
            double x = 0;
            double y = 0;
            double length_squared = 0;
            while (repeats < blk->iters && length_squared < 4)
            {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            };
            image[j * img_width + i] = repeats;
        }
    }
    delete blk;
}

int main(int argc, char **argv)
{
    /* argument parsing */
    assert(argc == 9);
    const char *filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    image_left = strtod(argv[3], 0);
    image_right = strtod(argv[4], 0);
    image_lower = strtod(argv[5], 0);
    image_upper = strtod(argv[6], 0);
    image_width = strtol(argv[7], 0, 10);
    image_height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    image = (int *)malloc(image_width * image_height * sizeof(int));
    assert(image);
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    const int n_cpu = CPU_COUNT(&cpu_set);

    /* mandelbrot set */
    Threadpool tp(n_cpu);
    const int grain_size = ((image_height / n_cpu) >> 10) + 1;
#ifdef DEBUG
    printf("grain_size=%d\n", grain_size);
#endif // DEBUG
    for (int y = 0; y < image_height; y += grain_size)
    {
        Task task;
        task.func = render;
        if (y + grain_size > image_height)
            task.args = new Block(0, image_width, y, image_height, iters);
        else
            task.args = new Block(0, image_width, y, y + grain_size, iters);
#ifdef DEBUG
        Block *blk = (Block *)task.args;
        printf("Submit %d, %d, %d, %d\n", blk->h0, blk->h1, blk->v0, blk->v1);
#endif // DEBUG
        tp.submit_task(task);
    }
    tp.run_task();

    /* draw and cleanup */
    write_png(filename, iters, image_width, image_height);
    free(image);
}