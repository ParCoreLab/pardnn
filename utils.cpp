#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include "utils.h"
using namespace std;

Folder_Files folder_files;

int K = 2;
bool BALANCE_MEMORY = false;
int REPLICATION_FACTOR = 0;
long long int DEVICE_MEM_CAPACITY = 32000000000l;
double COMM_LATENCY = 25.0;
double COMM_TRANSFER_RATE_RECIPROCAL = 1.0 / 130000;

void clean_line(string &line){
  string result = "";
  for(size_t i = 0; i < line.size(); i++){
    if(line[i] != '\n' and line[i] != '\r' and line[i] != '\t' and line[i] != ' ' and line[i] != '"'){
      result += line[i];
    }
  }
  line = result;
}

void clean_line_keep_spaces(string &line){
  string result = "";
  for(size_t i = 0; i < line.size(); i++){
    if(line[i] != '\n' and line[i] != '\r' and line[i] != '\t' and line[i] != '"'){
      result += line[i];
    }
  }
  line = result;
}

void print_vector(vector<string> vect){
  cout<<"[ ";
  for(int i=0; i < vect.size(); i++){
    cout<<vect[i]<<" ";
  }
  cout<<"]\n";
}

vector<string> split(const string& str, string delimiter)
{
  vector<string> tokens;
  string token;
  int indx = 0;
  size_t i = 0;
  string maybe = "";
  while(i < str.size()){
    while (str[i] == delimiter[indx]){
      maybe += str[i];
      indx++;
      i++;
      if(indx == delimiter.size()){
        maybe = "";
        tokens.push_back(token);
        token = "";
        break;
      }
    }
    indx = 0;
    if(maybe != ""){
      token += maybe;
      maybe = "";
    }
    token += str[i];
    i++;
  }
  if(token != ""){
    tokens.push_back(token);
  }
  return tokens;
}

void copy_unordered_map(unordered_map<string, int>& src, unordered_map<string, int>& dst){
  for(auto& it: src){
    dst[it.first] = it.second;
  }
}

void read_settings(){
  ifstream graph_file;
  string line;
  graph_file.open("settings.txt");
  while ( getline (graph_file,line) ){
    clean_line(line);
    vector<string> splits = split(line, "::");
    if(splits.size() == 2){
      if(splits[0] == "folder"){
        folder_files.folder = splits[1];
      }
      if(splits[0] == "graph"){
        folder_files.files[GRAPH_FILE] = folder_files.folder + "/" + splits[1];
      }
      else if(splits[0] == "nodes_weights"){
        folder_files.files[NODES_WEIGHTS_FILE] = folder_files.folder + "/" + splits[1];
      }
      else if(splits[0] == "edges_costs"){
        folder_files.files[EDGES_COSTS_FILE] = folder_files.folder + "/" + splits[1];
      }
      else if(splits[0] == "var_nodes"){
        folder_files.files[VAR_NODES_FILE] = folder_files.folder + "/" + splits[1];
      }
      else if(splits[0] == "ref_nodes"){
        folder_files.files[REF_NODES_FILE] = folder_files.folder + "/" + splits[1];
      }
      else if(splits[0] == "no_ops"){
        folder_files.files[NO_OPs_FILE] = folder_files.folder + "/" + splits[1];
      }
      else if(splits[0] == "cpu_nodes"){
        folder_files.files[CPU_NODES_FILE] = folder_files.folder + "/" + splits[1];
      }
      else if(splits[0] == "vanilla_cleaned"){
        folder_files.files[VANILLA_PLACEMENT_FILE] = folder_files.folder + "/" + splits[1];
      }
      else if(splits[0] == "memory_file"){
        folder_files.files[MEMORY_FILE_NAME] = folder_files.folder + "/" + splits[1];
      }
      else if(splits[0] == "K"){
        K = stoi(splits[1]);
      }
      else if(splits[0] == "COMM_TRANSFER_RATE"){
        COMM_TRANSFER_RATE_RECIPROCAL =  1.0 / ( stod(splits[1]) * 1000 );
      }
      else if(splits[0] == "COMM_LAENCY"){
        COMM_LATENCY = stod(splits[1]);
      }
      else if(splits[0] == "BALANCE_MEMORY"){
        BALANCE_MEMORY = (bool)stoi(splits[1]);
      }
      else if(splits[0] == "REPLICATION_FACTOR"){
        REPLICATION_FACTOR = stoi(splits[1]);
      }
      else if(splits[0] == "DEVICE_MEM_CAPACITY"){
        DEVICE_MEM_CAPACITY = stol(splits[1]);
      }
    }
  } 
}

/*         n --> No. of elements present in input array.  
    BITree[0..n] --> Array that represents Binary Indexed Tree. 
    arr[0..n-1] --> Input array for which prefix sum is evaluated. */
  
// Returns sum of arr[0..index]. This function assumes 
// that the array is preprocessed and partial sums of 
// array elements are stored in BITree[]. 
int getSum(int BITree[], int index) 
{ 
    int sum = 0; // Iniialize result 
  
    // index in BITree[] is 1 more than the index in arr[] 
    index = index + 1; 
  
    // Traverse ancestors of BITree[index] 
    while (index>0) 
    { 
        // Add current element of BITree to sum 
        sum += BITree[index]; 
  
        // Move index to parent node in getSum View 
        index -= index & (-index); 
    } 
    return sum; 
} 
  
// Updates a node in Binary Index Tree (BITree) at given index 
// in BITree. The given value 'val' is added to BITree[i] and  
// all of its ancestors in tree. 
void updateBIT(int BITree[], int n, int index, int val) 
{ 
    // index in BITree[] is 1 more than the index in arr[] 
    index = index + 1; 
  
    // Traverse all ancestors and add 'val' 
    while (index <= n) 
    { 
    // Add 'val' to current node of BI Tree 
    BITree[index] += val; 
  
    // Update index to that of parent in update View 
    index += index & (-index); 
    } 
} 
  
// Constructs and returns a Binary Indexed Tree for given 
// array of size n. 
int *constructBITree(int arr[], int n) 
{ 
    // Create and initialize BITree[] as 0 
    int *BITree = new int[n+1]; 
    for (int i=1; i<=n; i++) 
        BITree[i] = 0; 
  
    // Store the actual values in BITree[] using update() 
    for (int i=0; i<n; i++) 
        updateBIT(BITree, n, i, arr[i]); 
  
    // Uncomment below lines to see contents of BITree[] 
    //for (int i=1; i<=n; i++) 
    //     cout << BITree[i] << " "; 
  
    return BITree; 
} 

// Constructs and returns a Binary Indexed Tree for given 
// array of size n. 
int *constructBITree(map<int, int>& mp, int n) 
{ 
    // Create and initialize BITree[] as 0 
    int *BITree = new int[n+1]; 
    for (int i=1; i<=n; i++) 
        BITree[i] = 0; 
  
    // Store the actual values in BITree[] using update() 
    int i = 0;
    for (auto &it : mp) {
        updateBIT(BITree, n, i, it.second); 
        i++;
    }
    // Uncomment below lines to see contents of BITree[] 
    //for (int i=1; i<=n; i++) 
    //     cout << BITree[i] << " "; 
  
    return BITree; 
} 