#include "mapreduce.h"
// #include "pool.h"
#include "thread_pool.h"
#include <cassert>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <dirent.h>
#include <sys/stat.h>
#include <mpi.h>
#include <omp.h>
#include <unistd.h>
using namespace std;

#define INTERMEDIATE_FILE_EXTENSION ("intermediate")

void split_pre_process(const char *input_filename, int chunk_size, vector<MapReduce::Chunk> &chunks)
{
    ifstream ifs(input_filename);
    assert(ifs);

    string line;
    int line_number = 0;
    while (1)
    {
        MapReduce::Chunk chunk(ifs.tellg(), line_number, chunk_size);
        for (int i = 0; i < chunk_size; ++i)
            getline(ifs, line);

        if (ifs.eof())
            break;

        chunks.push_back(chunk);
        line_number += chunk_size;
    }
}

void *map_phase(void *arg)
{
    MapReduce::MapPhaseArgs *args = static_cast<MapReduce::MapPhaseArgs *>(arg);

    vector<MapReduce::Record> records;
    MapReduce::split(args->input_filename, args->chunk, records);

    vector<MapReduce::WordCount> pairs;
    for (const auto &record : records)
        MapReduce::map(record, pairs);

    // Store the index of pairs corresponding to the reducerID
    vector<vector<int>> reducer_buck(args->reducer_number);
    for (int idx = pairs.size() - 1; idx >= 0; --idx)
    {
        int reducer_id;
        MapReduce::partition(pairs[idx].word, args->reducer_number, reducer_id);
        reducer_buck[reducer_id].push_back(idx);
    }

    // write imtermediate files
    ofstream ofs;
    stringstream path;
    for (int reducer_id = 0; reducer_id < args->reducer_number; ++reducer_id)
    {
        // intermediate files naming rule: <job_name>_<#line>_<reducerID>.INTERMEDIATE_FILE_EXTENSION
        path << args->job_name << "_" << args->chunk.line_number << "_" << reducer_id << "." << INTERMEDIATE_FILE_EXTENSION;
        ofs.open(path.str());
        assert(ofs);

        for (auto idx : reducer_buck[reducer_id])
            ofs << pairs[idx].word << " " << pairs[idx].count << "\n";
        ofs.flush();
        ofs.close();
        path.str("");
    }

    return nullptr;
}

void *reduce_phase(void *arg)
{
    MapReduce::ReducePhaseArgs *args = static_cast<MapReduce::ReducePhaseArgs *>(arg);
    vector<MapReduce::WordCount> pairs;
    vector<MapReduce::WordCountGroup> group_pairs;

    // read imtermediate files
    ifstream ifs;
    DIR *dir = nullptr;
    dirent *entry = nullptr;
    dir = opendir("./");
    if (dir != nullptr)
    {
        while ((entry = readdir(dir)))
        {
            if (entry->d_type != DT_REG)
                continue;

            // intermediate files naming rule: <job_name>_<chunkID>_<reducerID>.INTERMEDIATE_FILE_EXTENSION
            string filename(entry->d_name);
            if (filename.rfind(INTERMEDIATE_FILE_EXTENSION) == string::npos)
                continue;

            // extract reducerID
            string temp(
                filename.begin() + filename.rfind("_") + 1,
                filename.begin() + filename.rfind(INTERMEDIATE_FILE_EXTENSION) - 1);
            if (args->reducer_id != stoi(temp))
                continue;

            // parse word counts
            MapReduce::WordCount pair;
            ifs.open(filename);
            assert(ifs);
            while (ifs >> pair.word >> pair.count)
                pairs.push_back(pair);
            ifs.close();
        }
    }

    MapReduce::sort(pairs);
    MapReduce::group(pairs, group_pairs);
    pairs.clear();
    MapReduce::reduce(group_pairs, pairs);
    MapReduce::write_word_count(args->output_dir, args->job_name, args->reducer_id, pairs);

    return nullptr;
}

void clean_intermediate_files(const char *dir_path)
{
    DIR *dir = nullptr;
    dirent *entry = nullptr;

    dir = opendir(dir_path);
    if (dir != nullptr)
    {
        while ((entry = readdir(dir)))
        {
            if (entry->d_type != DT_REG)
                continue;

            string filename(entry->d_name);
            stringstream path;
            if (filename.rfind(INTERMEDIATE_FILE_EXTENSION) != string::npos)
            {
                path << "./" << dir_path << "./" << filename;
                if (remove(path.str().c_str()) != 0)
                    fprintf(stderr, "Error deleting %s\n", path.str().c_str());
                else
                {
                    // printf("Delete %s\n", path.str().c_str()); // for debug
                }
                path.clear();
            }
        }
    }
}

int main(int argc, char **argv)
{
    /* arguments */
    assert(argc == 8);
    const char *job_name = argv[1];
    const int num_reducer = stoi(argv[2]);
    const int delay_sec = stoi(argv[3]);
    const char *input_filename = argv[4];
    const int chunk_size = stoi(argv[5]);
    const char *locality_config_filename = argv[6];
    const char *output_dir = argv[7];

    /* check path */
    struct stat output_dir_stat;
    stat(output_dir, &output_dir_stat);
    assert(output_dir_stat.st_mode & S_IFDIR);

    /* MPI arguments */
    int mpi_rank, mpi_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    const int jobtracker = mpi_size - 1;

    /* thread pool arguments */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    const int n_cpu = CPU_COUNT(&cpu_set);

    /* hadoop - word count */
    MPI_Request req;
    MPI_Status status;
    vector<MapReduce::Chunk> chunks;
    if (mpi_rank == jobtracker)
    {
        // map phase
        split_pre_process(input_filename, chunk_size, chunks);
        ifstream ifs(locality_config_filename);
        assert(ifs);

        int chunk_id, node_id, non_local_cnt = 0;
        while (ifs >> chunk_id >> node_id)
        {
            MapReduce::Chunk &chunk = chunks[chunk_id - 1];
            int chunk_buf[4] = {chunk.streamoff, chunk.line_number, chunk.size};
            chunk_id %= mpi_size;
            node_id %= mpi_size - 1;
            chunk_buf[3] = (chunk_id == node_id) ? MapReduce::LOCAL_CHUNK : MapReduce::NON_LOCAL_CHUNK;
            non_local_cnt += int(chunk_id == node_id);
            // printf("%d SEND CHUNK(%d, %d, %d) to %d\n", jobtracker, chunk_buf[0], chunk_buf[1], chunk_buf[2], node_id); // for debug
            MPI_Isend(&chunk_buf, 4, MPI_INT, node_id, MapReduce::INT_TAG, MPI_COMM_WORLD, &req);
            MPI_Wait(&req, MPI_STATUS_IGNORE);
        }
        printf("#Non-Local=%d\n", non_local_cnt);
        for (int node_id = 0; node_id < mpi_size - 1; ++node_id)
        {
            int chunk_buf[3] = {MapReduce::FINISH_SPLIT, 0, 0};
            // printf("%d SEND FINISH SPLIT(%d, %d, %d) to %d\n", jobtracker, chunk_buf[0], chunk_buf[1], chunk_buf[2], node_id); // for debug
            MPI_Isend(&chunk_buf, 3, MPI_INT, node_id, MapReduce::INT_TAG, MPI_COMM_WORLD, &req);
            MPI_Wait(&req, MPI_STATUS_IGNORE);
        }
        for (int node_id = 0; node_id < mpi_size - 1; ++node_id)
        {
            int finish;
            MPI_Recv(&finish, 1, MPI_INT, node_id, MapReduce::INT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            assert(finish == MapReduce::FINISH_MAP);
        }

        // reduce phase
        if (num_reducer < mpi_size - 1)
        {
            for (int node_id = 0; node_id < num_reducer; ++node_id)
            {
                int reducer_id_range[2] = {node_id, node_id + 1};
                // printf("%d SEND RANGE(%d, %d) to %d\n", jobtracker, reducer_id_range[0], reducer_id_range[1], node_id); // for debug
                MPI_Isend(&reducer_id_range, 2, MPI_INT, node_id, MapReduce::INT_TAG, MPI_COMM_WORLD, &req);
                MPI_Wait(&req, MPI_STATUS_IGNORE);
            }
        }
        else
        {
            int range_size =
                (num_reducer % (mpi_size - 1))
                    ? num_reducer / (mpi_size - 1) + 1
                    : num_reducer / (mpi_size - 1);
            for (int node_id = 0; node_id < mpi_size - 1; ++node_id)
            {
                int reducer_id_range[2] = {node_id * range_size, node_id * range_size + range_size};
                if (node_id == mpi_size - 2) // last worker
                    reducer_id_range[1] = num_reducer;
                // printf("%d SEND RANGE(%d, %d) to %d\n", jobtracker, reducer_id_range[0], reducer_id_range[1], node_id); // for debug
                MPI_Isend(&reducer_id_range, 2, MPI_INT, node_id, MapReduce::INT_TAG, MPI_COMM_WORLD, &req);
                MPI_Wait(&req, MPI_STATUS_IGNORE);
            }
        }
        for (int node_id = 0; node_id < mpi_size - 1; ++node_id)
        {
            int finish;
            MPI_Recv(&finish, 1, MPI_INT, node_id, MapReduce::INT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            assert(finish == MapReduce::FINISH_REDUCE);
        }
    }
    else
    {
        MapReduce::ThreadPool *pool = nullptr;

        // map phase
        vector<MapReduce::Chunk> chunks;
        vector<int> chunk_states;
        while (1)
        {
            int chunk_buf[4];
            MPI_Recv(&chunk_buf, 4, MPI_INT, jobtracker, MapReduce::INT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (chunk_buf[0] != MapReduce::FINISH_SPLIT)
            {
                // printf("%d RECV CHUNK(%d, %d, %d)\n", mpi_rank, chunk_buf[0], chunk_buf[1], chunk_buf[2]); // for debug
                chunks.push_back(MapReduce::Chunk(chunk_buf[0], chunk_buf[1], chunk_buf[2]));
                chunk_states.push_back(chunk_buf[3]);
            }
            else
            {
                // printf("%d RECV FINISH SPLIT(%d, %d, %d)\n", mpi_rank, chunk_buf[0], chunk_buf[1], chunk_buf[2]); // for debug
                break;
            }
        }
        // printf("%d START MAP\n", mpi_rank); // for debug
        vector<MapReduce::MapPhaseArgs> map_task_args;
        for (int i = 0; i < chunks.size(); ++i)
        {
            map_task_args.push_back(
                MapReduce::MapPhaseArgs(
                    input_filename,
                    job_name, chunks[i],
                    num_reducer));
        }
#pragma omp parallel for schedule(dynamic) default(shared)
        for (int i = 0; i < map_task_args.size(); ++i)
        {
            if (chunk_states[i] == MapReduce::NON_LOCAL_CHUNK)
                sleep(delay_sec);
            map_phase((void *)(&map_task_args[i]));
        }
        /*
        pool = new MapReduce::ThreadPool(n_cpu - 1); // #mapper = #cpu - 1
        for (auto &arg : map_task_args)
        {
            MapReduce::ThreadPoolTask *task =
                new MapReduce::ThreadPoolTask(&map_phase, &arg);
            pool->addTask(task);
        }
        pool->start();
        while (pool->getNumTask() > 0)
        {
        }
        pool->terminate();
        pool->join();
        delete pool;
         */
        map_task_args.clear();
        MPI_Isend(&MapReduce::FINISH_MAP, 1, MPI_INT, jobtracker, MapReduce::INT_TAG, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        // printf("%d FINISH MAP\n", mpi_rank); // for debug

        // reduce phase
        int reducer_id_range[2];
        MPI_Recv(&reducer_id_range, 2, MPI_INT, jobtracker, MapReduce::INT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // printf("%d START REDUCE\n", mpi_rank); // for debug

        vector<MapReduce::ReducePhaseArgs> reduce_task_args;
        for (int reducer_id = reducer_id_range[0]; reducer_id < reducer_id_range[1]; ++reducer_id)
        {
            reduce_task_args.push_back(
                MapReduce::ReducePhaseArgs(
                    output_dir,
                    job_name,
                    reducer_id));
        }
#pragma omp parallel for schedule(dynamic) default(shared)
        for (auto &arg : reduce_task_args)
            reduce_phase((void *)&arg);
        /* pool = new MapReduce::ThreadPool(reducer_id_range[1] - reducer_id_range[0]);
        for (int reducer_id = reducer_id_range[0]; reducer_id < reducer_id_range[1]; ++reducer_id)
        {
            reduce_task_args.push_back(
                MapReduce::ReducePhaseArgs(
                    output_dir,
                    job_name,
                    reducer_id));
            MapReduce::ThreadPoolTask *task =
                new MapReduce::ThreadPoolTask(&reduce_phase, &(reduce_task_args.back()));
            pool->addTask(task);
        }
        pool->start();
        while (pool->getNumTask() > 0)
        {
        }
        pool->terminate();
        pool->join();
        delete pool; */
        reduce_task_args.clear();
        MPI_Isend(&MapReduce::FINISH_REDUCE, 1, MPI_INT, jobtracker, MapReduce::INT_TAG, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        // printf("%d FINISH REDUCE\n", mpi_rank); // for debug
    }

    // printf("%d FINISH\n", mpi_rank); // for debug
    MPI_Barrier(MPI_COMM_WORLD);
    if (mpi_rank == jobtracker)
        clean_intermediate_files("./");

    MPI_Finalize();
    return 0;
}