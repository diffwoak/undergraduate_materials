#include <iostream>
#include <vector>
#include <queue>
#include <algorithm> 
#include <omp.h>
#include <mpi.h>
#include <time.h>
using namespace std;

std::vector<std::vector<int> > generateRandomGraph(int numNodes, int avgDegree) {
    std::srand(time(0));
    std::vector<std::vector<int> > graph(numNodes);
    for (int i = 0; i < numNodes; ++i) {
        // 计算每个节点的邻居数量
        int numNeighbors = std::rand() % avgDegree + 1; // 邻居数量在 [0, 2 * avgDegree] 之间
        // 随机选择邻居节点
        for (int j = 0; j < numNeighbors; ++j) {
            int neighbor = std::rand() % numNodes;
            // 确保不连接到自己，并且没有重复的边
            if (neighbor != i && std::find(graph[i].begin(), graph[i].end(), neighbor) == graph[i].end()) {
                graph[i].push_back(neighbor);
                graph[neighbor].push_back(i);// 对称连接
            }
        }
    }
    return graph;
}

void printGraph(const std::vector<std::vector<int> >& graph) {
    for (int i = 0; i < graph.size(); ++i) {
        printf("Generate Node %d: ", i);
        for (int neighbor : graph[i]) {
            printf("%d ", neighbor);
            //  std::cout << neighbor << " ";
        }printf("\n");
        //std::cout << std::endl;
    }printf("\n");
}

void BFS(std::vector<std::vector<int> >& graph, int start) {
    int num_nodes = graph.size();
    std::vector<int> distance(num_nodes, -1);
    std::queue<int> bfs_queue;
    distance[start] = 0;
    bfs_queue.push(start);
    while (!bfs_queue.empty()) {
        int current_node = bfs_queue.front();
        bfs_queue.pop();
        for (int j : graph[current_node]) {
            if (distance[j] == -1) {
                distance[j] = distance[current_node] + 1;
                bfs_queue.push(j);
            }
        }
    }
    for (int i = 0; i < num_nodes; ++i) {
        //std::cout << "Distance from " << start << " to " << i << ": " << distance[i] << std::endl;
    }
}

void parallelBFS(std::vector<std::vector<int> >& graph, int start) {
    int num_nodes = graph.size();
    std::vector<int> distance(num_nodes, -1);
    std::queue<int> bfs_queue;
    int current_node;
    distance[start] = 0;
    bfs_queue.push(start);
    omp_set_num_threads(8);
    while (!bfs_queue.empty()) {
        int len = bfs_queue.size();
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < len; i++) {
#pragma omp critical
            {                        
                current_node = bfs_queue.front();
                bfs_queue.pop();
                //printf("working thread:%d node:%d\n", omp_get_thread_num(), current_node);
            }                    
            for (int j : graph[current_node]) {//并行遍历相邻节点
                if (distance[j] == -1) {
#pragma omp critical
                    {
                        //printf("     In thread:%d sub_node:%d\n", omp_get_thread_num(), j);
                        distance[j] = distance[current_node] + 1;
                        bfs_queue.push(j);
                    }

                }               
            }
        }
#pragma omp barrier
    }

    for (int i = 0; i < num_nodes; ++i) {
        //printf("Distance from %d to %d :%d\n", start, i, distance[i]);
    }
}

void mpiBFS(std::vector<std::vector<int> >& graph, int start, int id) {
    int num_nodes = graph.size();
    std::vector<int> distance(num_nodes, -1);
    std::vector<int> bfs_list, temp_list0, temp_list1;
    int localstart, localend;
    omp_set_num_threads(8);
    distance[start] = 0;
    bfs_list.push_back(start);
    while (!bfs_list.empty()) {
        if (id == 0)temp_list0.clear();
        else temp_list1.clear();
        int len = bfs_list.size();
        localstart = id * len / 2;
        localend = len / (2 - id);
#pragma omp parallel for schedule(dynamic)
        for (int i = localstart; i < localend; i++) {
            int current_node = bfs_list[i];
            for (int j : graph[current_node]) {
                if (distance[j] == -1) {
#pragma omp critical
                    {
                        printf("id:%d working thread:%d\n", id, omp_get_thread_num());
                        distance[j] = distance[current_node] + 1;
                        if (id == 0) temp_list0.push_back(j);
                        else temp_list1.push_back(j);
                    }
                }                 
            }
        }
#pragma omp barrier
        bfs_list.clear();
        if (id == 0) {
            int listsize, totlistsize;
            MPI_Recv(&listsize, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::vector<int> rece_distance(num_nodes), rece_list(listsize);
            MPI_Recv(rece_distance.data(), num_nodes, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(rece_list.data(), listsize, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int l0 : temp_list0) {
                bfs_list.push_back(l0);
            }
            for (int l1 : rece_list) {
                bfs_list.push_back(l1);
            }totlistsize = bfs_list.size();
            for (int i = 0; i < num_nodes; i++) {
                if (rece_distance[i] == -1);
                else if (distance[i] == -1)distance[i] = rece_distance[i];
                else distance[i] = distance[i] < rece_distance[i] ? distance[i] : rece_distance[i];
            }
            MPI_Send(distance.data(), num_nodes, MPI_INT, 1, 1, MPI_COMM_WORLD);
            MPI_Send(&totlistsize, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
            MPI_Send(bfs_list.data(), totlistsize, MPI_INT, 1, 1, MPI_COMM_WORLD);
        }
        else {
            int listsize = temp_list1.size(), totlistsize;
            MPI_Send(&listsize, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(distance.data(), num_nodes, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(temp_list1.data(), listsize, MPI_INT, 0, 1, MPI_COMM_WORLD);
            std::vector<int>  rec_distance(num_nodes);
            MPI_Recv(rec_distance.data(), num_nodes, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&totlistsize, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::vector<int>  rec_bfs_list(totlistsize);
            MPI_Recv(rec_bfs_list.data(), totlistsize, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i : rec_bfs_list) {
                bfs_list.push_back(i);
            }
            for (int i = 0; i < num_nodes; i++) {
                distance[i] = rec_distance[i]; 
            }
        }
    }
    if (id == 0) {
        for (int i = 0; i < num_nodes; ++i) {
            printf("Distance from %d to %d :%d\n", start, i, distance[i]);        
        }
    }

}

int main(int argc, char* argv[]) {
    int numNodes = 100;
    int avgDegree = 50;
    int id, size;
    std::vector<std::vector<int> > graph = generateRandomGraph(numNodes, avgDegree);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    //if (id == 0) {printGraph(graph);}
    /*
    if (id == 0) {
        double t_start1 = MPI_Wtime();
        BFS(graph, 0);
        double t_end1 = MPI_Wtime();
        printf("Total time: %f s\n", t_end1 - t_start1);
        double t_start2 = MPI_Wtime();
        parallelBFS(graph, 0);
        double t_end2 = MPI_Wtime();
        printf("parallel_8 total time: %f s\n", t_end2 - t_start2);
    }*/
    double t_start3 = MPI_Wtime();
    mpiBFS(graph, 0, id);
    double t_end3 = MPI_Wtime();
    printf("parallel_16 total time: %f s\n", t_end3 - t_start3);
    MPI_Finalize();
    return 0;
}