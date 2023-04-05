#include <mpi.h>
#include "mapreduce.h"
#include <fstream>
#include <sstream>
#include <cassert>
#include <algorithm>
#include <utility>
using namespace std;

namespace MapReduce
{
    /* Input split function: It reads a data chunk from the input file,
     * and splits it into a set of records. Each line in the input file
     * is considered as a record. The key of a record is its line number
     * in the input file, and the value of a record is the text in the line. */
    void split(
        const char *input_filename,
        const Chunk &chunk,
        vector<Record> &records)
    {
        ifstream ifs(input_filename);
        assert(ifs);
        ifs.seekg(chunk.streamoff, ios::beg);

        string line;
        for (int i = 0; i < chunk.size && getline(ifs, line); ++i)
        {
            records.push_back(Record(chunk.line_number + i, line));
            // records.push_back(Record(chunk.line_number + i, move(line)));
        }
    }
    void *split(void *arg)
    {
        split(
            static_cast<InputSplitArgs *>(arg)->input_filename,
            static_cast<InputSplitArgs *>(arg)->chunk,
            static_cast<InputSplitArgs *>(arg)->records);
        return nullptr;
    }

    /* Map function: It reads an input key-value pair record and output to a set
     * of intermediate key-value pairs. You can assume the data type of the input
     * records is int (i.e., line#), and string (i.e., line text), and the data
     * type of the output records is string (i.e.,word), and int (i.e., count). */
    void map(
        const Record &record,
        vector<WordCount> &pairs)
    {
        stringstream ss(record.text);

        string word;
        while (ss >> word)
        {
            pairs.push_back(WordCount(word, 1));
            // pairs.push_back(WordCount(move(word), 1));
        }
    }
    void *map(void *arg)
    {
        map(
            static_cast<MapArgs *>(arg)->record,
            static_cast<MapArgs *>(arg)->pairs);
        return nullptr;
    }

    /* Partition function: It is a hash function that maps keys to reducers. The
     * output should be the reducerID, hence the return value should be bounded
     * by the number of reducers. */
    void partition(
        const string &word,
        const int reducer_number,
        int &reducer_id)
    {
        reducer_id = 0;
        for (const auto ch : word)
            reducer_id += int(ch);
        reducer_id %= reducer_number;
    }
    void *partition(void *arg)
    {
        partition(
            static_cast<PartitionArgs *>(arg)->word,
            static_cast<PartitionArgs *>(arg)->reducer_number,
            static_cast<PartitionArgs *>(arg)->reducer_id);
        return nullptr;
    }

    /* Sort function: It sorts the keys before passing them to the reducers. The
     * default implementation should follow the ascending order of ASCII code. */
    void sort(vector<WordCount> &pairs)
    {
        std::sort(
            pairs.begin(),
            pairs.end(),
            [](const WordCount &p1, const WordCount &p2)
            {
                return p1.word < p2.word;
            });
    }
    void *sort(void *arg)
    {
        sort(static_cast<SortArgs *>(arg)->pairs);
        return nullptr;
    }

    /* Group function: It contains a comparison function of the keys to determine
     * what are the values that should be grouped together for calling a reduce
     * function. The default implementation should only group the values with the
     * exact same key together. */
    void group(
        vector<WordCount> &pairs,
        vector<WordCountGroup> &group_pairs)
    {
        string last_word;
        for (auto &pair : pairs)
        {
            if (pair.word != last_word)
            {
                last_word = pair.word;
                group_pairs.push_back(WordCountGroup(pair.word, pair.count));
                // group_pairs.push_back(WordCountGroup(move(pair.word), pair.count));
            }
            else
            {
                group_pairs.back().counts.push_back(pair.count);
            }
        }
    }
    void *group(void *arg)
    {
        group(
            static_cast<GroupArgs *>(arg)->pairs,
            static_cast<GroupArgs *>(arg)->group_pairs);
        return nullptr;
    }

    /* Reduce function: It aggregates the values of a key, and outputs its final
     * key-value pairs. You can assume the data type of the output key-value pair
     * is string (i.e.,word), and int (i.e., count). */
    void reduce(
        vector<WordCountGroup> &group_pairs,
        vector<WordCount> &pairs)
    {
        for (auto &group : group_pairs)
        {
            int total = 0;
            for (auto &count : group.counts)
                total += count;
            pairs.push_back(WordCount(group.word, total));
            // pairs.push_back(WordCount(move(group.word), total));
        }
    }
    void *reduce(void *arg)
    {
        reduce(
            static_cast<ReduceArgs *>(arg)->group_pairs,
            static_cast<ReduceArgs *>(arg)->pairs);
        return nullptr;
    }

    /* Output function: It writes all the output key-value pairs of a reduce task
     * thread to an output file stored on NFS. The default implementation for the
     * output format should be one key per line, and separate the key and values
     * by space. */
    void write_word_count(
        const char *output_dir,
        const char *job_name,
        const int reducer_id,
        const vector<WordCount> &pairs)
    {
        stringstream path;
        path << "./" << output_dir << "./" << job_name << "-" << reducer_id << ".out";
        ofstream ofs(path.str());
        assert(ofs);

        for (const auto &pair : pairs)
            ofs << pair.word << " " << pair.count << "\n";

        ofs.flush();
        ofs.close();
    }
    void *write_word_count(void *arg)
    {
        write_word_count(
            static_cast<OutputArgs *>(arg)->output_dir,
            static_cast<OutputArgs *>(arg)->job_name,
            static_cast<OutputArgs *>(arg)->reducer_id,
            static_cast<OutputArgs *>(arg)->pairs);
        return nullptr;
    }
};