// usage: ./verifier <answer_file_path> <result_dir_path>
#include <fstream>
#include <unordered_map>
#include <cassert>
#include <dirent.h>
using namespace std;

void append(const char *filename, unordered_map<string, int> &map)
{
    printf("%s\n", filename);
    ifstream ifs(filename);
    if (!ifs)
    {
        printf("Skip!\n");
        return;
    }

    string word;
    int count;

    while (ifs >> word >> count)
    {
        auto iter = map.find(word);
        if (iter != map.end())
            iter->second += count;
        else
            map.insert(make_pair(word, count));
    }
}

int main(int argc, char **argv)
{
    const char *ans_file = argv[1];
    const char *output_dir = argv[2];
    string extension = argc == 4 ? ".intermediate" : ".out";

    unordered_map<string, int> ans, result;
    append(ans_file, ans);

    DIR *dir = nullptr;
    dirent *entry = nullptr;

    dir = opendir(output_dir);
    if (dir != nullptr)
    {
        while ((entry = readdir(dir)))
        {
            if (entry->d_type != DT_REG)
                continue;

            string path = "./" + string(output_dir) + "./" + entry->d_name;
            if (path.rfind(extension) != string::npos)
                append(path.c_str(), result);
        }
    }

    if (ans == result)
        printf("Correct!\n");
    else
        printf("Not correct.\n");

    return 0;
}