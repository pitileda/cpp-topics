#pragma once

class My
{
private:
	int num_;
public:
	My();
	~My();
};

class Derived : public My
{
private:
	bool der_;
public:
	Derived();
	~Derived();
};

int calculate();