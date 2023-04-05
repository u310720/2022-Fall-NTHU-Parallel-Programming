#pragma once

#include <string>
#include <vector>

namespace MapReduce
{
    static const int LOCAL_CHUNK = -5;
    static const int NON_LOCAL_CHUNK = -4;
    static const int FINISH_SPLIT = -3;
    static const int FINISH_MAP = -2;
    static const int FINISH_REDUCE = -1;
    static const int INT_TAG = 1;
    static const int STR_TAG = 2;
    static const int LONG_TAG = 3;

    struct Chunk
    {
        /* args for read file */
        int streamoff; // move pointer to ios::beg + streamoff

        /* args for Record */
        int line_number;
        int size;

        Chunk(int streamoff, int line_number, int size)
            : streamoff(streamoff),
              line_number(line_number),
              size(size) {}
    };

    struct Record
    {
        int line_number;  // key
        std::string text; // value

        Record(int line_number, const std::string &text) : line_number(line_number), text(text) {}
        Record(int line_number, std::string &&text) : line_number(line_number), text(text) {}
    };

    struct WordCount
    {
        std::string word;
        int count;

        WordCount() = default;
        WordCount(const std::string &word, int count) : word(word), count(count) {}
        WordCount(std::string &&word, int count) : word(word), count(count) {}
    };

    struct WordCountGroup
    {
        std::string word;
        std::vector<int> counts;

        WordCountGroup(const std::string &word, int count) : word(word), counts(1, count) {}
        WordCountGroup(std::string &&word, int count) : word(word), counts(1, count) {}
    };

    struct MapPhaseArgs
    {
        const char *input_filename;
        const char *job_name;
        const Chunk &chunk;
        const int reducer_number;

        MapPhaseArgs(const char *input_filename,
                     const char *job_name,
                     const Chunk &chunk,
                     int reducer_number)
            : input_filename(input_filename),
              job_name(job_name),
              chunk(chunk),
              reducer_number(reducer_number) {}
    };

    struct ReducePhaseArgs
    {
        const char *output_dir;
        const char *job_name;
        const int reducer_id;

        ReducePhaseArgs(
            const char *output_dir,
            const char *job_name,
            int reducer_id)
            : output_dir(output_dir),
              job_name(job_name),
              reducer_id(reducer_id) {}
    };

    /* Input split function: It reads a data chunk from the input file,
     * and splits it into a set of records. Each line in the input file
     * is considered as a record. The key of a record is its line number
     * in the input file, and the value of a record is the text in the line. */
    struct InputSplitArgs
    {
        const char *input_filename;
        const Chunk &chunk;
        std::vector<Record> &records;

        InputSplitArgs(
            const char *input_filename,
            const Chunk &chunk,
            std::vector<Record> &records)
            : input_filename(input_filename),
              chunk(chunk),
              records(records) {}
    };
    void *split(void *arg);
    void split(
        const char *input_filename,  // input
        const Chunk &chunk,          // input
        std::vector<Record> &records // output
    );

    /* Map function: It reads an input key-value pair record and output to a set
     * of intermediate key-value pairs. You can assume the data type of the input
     * records is int (i.e., line#), and string (i.e., line text), and the data
     * type of the output records is string (i.e.,word), and int (i.e., count). */
    struct MapArgs
    {
        const Record &record;
        std::vector<WordCount> &pairs;

        MapArgs(
            const Record &record,
            std::vector<WordCount> &pairs)
            : record(record),
              pairs(pairs) {}
    };
    void *map(void *arg);
    void map(
        const Record &record,         // input
        std::vector<WordCount> &pairs // output
    );

    /* Partition function: It is a hash function that maps keys to reducers. The
     * output should be the reducerID, hence the return value should be bounded
     * by the number of reducers. */
    struct PartitionArgs
    {
        const std::string &word;
        int reducer_number;
        int &reducer_id;

        PartitionArgs(
            const std::string &word,
            int reducer_number,
            int &reducer_id)
            : word(word),
              reducer_number(reducer_number),
              reducer_id(reducer_id) {}
    };
    void *partition(void *arg);
    void partition(
        const std::string &word,  // input
        const int reducer_number, // input
        int &reducer_id           // output
    );

    /* Sort function: It sorts the keys before passing them to the reducers. The
     * default implementation should follow the ascending order of ASCII code. */
    struct SortArgs
    {
        std::vector<WordCount> &pairs;
        SortArgs(std::vector<WordCount> &pairs) : pairs(pairs) {}
    };
    void *sort(void *arg);
    void sort(std::vector<WordCount> &pairs);

    /* Group function: It contains a comparison function of the keys to determine
     * what are the values that should be grouped together for calling a reduce
     * function. The default implementation should only group the values with the
     * exact same key together. */
    struct GroupArgs
    {
        std::vector<WordCount> &pairs;
        std::vector<WordCountGroup> &group_pairs;

        GroupArgs(
            std::vector<WordCount> &pairs,
            std::vector<WordCountGroup> &group_pairs)
            : pairs(pairs),
              group_pairs(group_pairs) {}
    };
    void *group(void *arg);
    void group(
        std::vector<WordCount> &pairs,           // input
        std::vector<WordCountGroup> &group_pairs // output
    );

    /* Reduce function: It aggregates the values of a key, and outputs its final
     * key-value pairs. You can assume the data type of the output key-value pair
     * is string (i.e.,word), and int (i.e., count). */
    struct ReduceArgs
    {
        std::vector<WordCountGroup> &group_pairs;
        std::vector<WordCount> &pairs;

        ReduceArgs(
            std::vector<WordCountGroup> &group_pairs,
            std::vector<WordCount> &pairs)
            : group_pairs(group_pairs),
              pairs(pairs) {}
    };
    void *reduce(void *arg);
    void reduce(
        std::vector<WordCountGroup> &group_pairs, // input
        std::vector<WordCount> &pairs             // output
    );

    /* Output function: It writes all the output key-value pairs of a reduce task
     * thread to an output file stored on NFS. The default implementation for the
     * output format should be one key per line, and separate the key and values
     * by space. */
    struct OutputArgs
    {
        const char *output_dir;
        const char *job_name;
        int reducer_id;
        std::vector<WordCount> pairs;

        OutputArgs(
            const char *output_dir,
            const char *job_name,
            int reducer_id,
            std::vector<WordCount> &pairs)
            : output_dir(output_dir),
              job_name(job_name),
              reducer_id(reducer_id),
              pairs(pairs) {}
    };
    void *write_word_count(void *arg);
    void write_word_count(
        const char *output_dir,             // input
        const char *job_name,               // input
        const int reducer_id,               // input
        const std::vector<WordCount> &pairs // input
    );
};