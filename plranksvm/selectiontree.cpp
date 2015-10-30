#include "selectiontree.h"
selectiontree::selectiontree(int l)
{
	int i = 1;
	while(i < l)
		i *= 2;
	this->leaf_size = l;
	this->node_size = i-1;
	node = new tree_node[i+l];
	for (int j=0;j<i+l;j++)
	{
		node[j].size=0;
		node[j].vector_sum=0;
	}
}

selectiontree::~selectiontree()
{
	delete[] node;
}

void selectiontree::insert_node(int key, double value)
{
	key += this->node_size;
	while(key > 0)
	{
		node[key].vector_sum += value;
		node[key].size++;
		key /= 2;
	}
}

void selectiontree::count_larger(int key, int* count_ret, double* acc_value_ret)
{
	if (key >= this->leaf_size)
	{
		*count_ret = 0;
		*acc_value_ret = 0;
		return;
	}
	int count = 0;
	double acc_value = 0;
	key += node_size;
	while(key > 1)
	{
		if (key % 2 ==0)
		{
			count += node[key+1].size;
			acc_value += node[key+1].vector_sum;
		}
		key /= 2;
	}
	*count_ret = count;
	*acc_value_ret = acc_value;
}

void selectiontree::count_smaller(int key, int* count_ret, double* acc_value_ret)
{
	if (key <= 1)
	{
		*count_ret = 0;
		*acc_value_ret = 0;
		return;
	}
	int count = 0;
	double acc_value = 0;
	key += node_size;
	while(key > 1)
	{
		if (key%2)
		{
			count += node[key-1].size;
			acc_value += node[key-1].vector_sum;
		}
		key /= 2;
	}
	*count_ret = count;
	*acc_value_ret = acc_value;
}

double selectiontree::vector_sum_larger(int key)
{
	if (key >= this->leaf_size)
		return 0;
	double acc_value = 0;
	key += node_size;
	while(key > 1)
	{
		if (key%2 ==0)
		{
			acc_value += node[key+1].vector_sum;
		}
		key /= 2;
	}
	return acc_value;
}

double selectiontree::vector_sum_smaller(int key)
{
	if (key <= 1)
		return 0;
	double acc_value = 0;
	key+=node_size;
	while(key > 1)
	{
		if (key%2)
		{
			acc_value += node[key-1].vector_sum;
		}
		key /= 2;
	}
	return acc_value;
}

