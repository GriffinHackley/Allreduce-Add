#include <iostream>
#include <mpi.h>
#include <unistd.h>
#include <stdlib.h>
#include <numeric>
#include <math.h>

#define MCW MPI_COMM_WORLD

using namespace std;

void allReduce(){
    int rank, data, size, value = 0;
    MPI_Comm_rank(MCW, &rank);

    data = rank;
    MPI_Allreduce(&data, &value, 1, MPI_INT, MPI_SUM, MCW);

    //print one
    if(rank == 0){
        cout << "Allreduce : " << value << endl;
    }

    //print all
    // cout << "process " << rank << " got " << value << endl;
}

void gather(){
    //initialize
    MPI_Barrier(MCW);
    int rank, data, size, value, result = 0;
    MPI_Comm_rank(MCW, &rank);
    MPI_Comm_size(MCW, &size);
    int recvData[size];
    

    //gather rank from every process
    data = rank;
    MPI_Gather(&data,1,MPI_INT,recvData,1,MPI_INT,0,MCW);

    //add all of the gathered data together
    if(rank == 0){
        for(int i = 0; i < size; i++){
            value += recvData[i];
        }
        cout << "gather    : " << value << endl;
    }

    // //broadcast result to all 
    data = value;
    MPI_Bcast(&data,1,MPI_INT,0,MCW);
    MPI_Barrier(MCW);

    //print from all
    // cout << "process " << rank << " got " << data << endl;
}

void leader(){
    MPI_Barrier(MCW);
    int rank, data, size, value = 0;
    MPI_Comm_rank(MCW, &rank);
    MPI_Comm_size(MCW, &size);

    if(rank == 0){
        //get messages from all processes and add them together
        for(int i = 0; i < size-1; i++){
            MPI_Recv(&data, 1, MPI_INT, MPI_ANY_SOURCE, 0, MCW, MPI_STATUS_IGNORE);
            value += data;
        }

        //send result to other processes
        data = value;
        for(int i = 1; i < size; i++){
            MPI_Send(&data, 1, MPI_INT, i, 0, MCW);
        }

        //print result
        cout << "leader    : " << value << endl;

    } else {
        //if not rank 0 send rank to process 0
        data = rank;
        MPI_Send(&data, 1, MPI_INT, 0, 0, MCW);

        //recieve result
        MPI_Recv(&data, 1, MPI_INT, MPI_ANY_SOURCE, 0, MCW, MPI_STATUS_IGNORE);
    }
    //print from all
    // cout << "process " << rank << " got " << data << endl;
}

void ring(){
    MPI_Barrier(MCW);
    int rank, size, data;
    MPI_Comm_rank(MCW, &rank);
    MPI_Comm_size(MCW, &size);

    //start at rank 0 and send data to next process
    if(rank == 0){
        data = rank;
        MPI_Send(&data, 1, MPI_INT, (rank+1)%size, 0, MCW);
        MPI_Recv(&data, 1, MPI_INT, MPI_ANY_SOURCE, 0, MCW, MPI_STATUS_IGNORE);

        //after 0 has recieved a message from the last process, print the results
        cout << "ring      : " << data << endl;
    } else {
        //once you have recieved the data send to the next one
        MPI_Recv(&data, 1, MPI_INT, MPI_ANY_SOURCE, 0, MCW, MPI_STATUS_IGNORE);
        data += rank;
        MPI_Send(&data, 1, MPI_INT, (rank+1)%size, 0, MCW);
    }

    //send result to all
    if(rank == 0){
        MPI_Send(&data, 1, MPI_INT, (rank+1)%size, 0, MCW);
        MPI_Recv(&data, 1, MPI_INT, MPI_ANY_SOURCE, 0, MCW, MPI_STATUS_IGNORE);
    } else {
        //once you have recieved the data send to the next one
        MPI_Recv(&data, 1, MPI_INT, MPI_ANY_SOURCE, 0, MCW, MPI_STATUS_IGNORE);
        MPI_Send(&data, 1, MPI_INT, (rank+1)%size, 0, MCW);
    }
    //print from all
    // cout << "process " << rank << " got " << data << endl;
}

void hyperCube(){
    MPI_Barrier(MCW);
    int rank, data, size, value, dest, times = 0;
    unsigned int mask = 1;
    MPI_Comm_rank(MCW, &rank);
    MPI_Comm_size(MCW, &size);

    value = rank;

    //add up across each axis
    for(int i = 0; i < log2(size); i++){
        dest = rank^(mask<<i);
        data = value;
        MPI_Send(&data, 1, MPI_INT, dest, 0, MCW);
        MPI_Recv(&data, 1, MPI_INT, MPI_ANY_SOURCE, 0, MCW, MPI_STATUS_IGNORE);
        value += data;
        MPI_Barrier(MCW);
    }

    //print only one value
    if(rank == 0){
        cout << "hypercube : " << value << endl;
    }
    //print from all
    // cout << "process " << rank << " got " << value << endl;
}

int main(int argc, char **argv){
    int size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MCW, &size);
    

    //run all
    gather();
    allReduce();
    leader();
    ring();
    hyperCube();

	MPI_Finalize();
	return 0;
}