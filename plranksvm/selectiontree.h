struct tree_node
{
	double vector_sum;
	int size;
};

class selectiontree
{
public:
	selectiontree(int l);
	~selectiontree();
	void insert_node(int key, double value);
	void delete_node(int key, double value);
	void count_larger(int key, int* count_ret, double* acc_value_ret);
	void count_smaller(int key, int* count_ret, double* acc_value_ret);
	double vector_sum_larger(int key);
	double vector_sum_smaller(int key);
private:
	int node_size;
	int leaf_size;
	tree_node* node;
};

