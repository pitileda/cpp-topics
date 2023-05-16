#include <iostream>
#include <memory>

class IMethod {
public:
    virtual void ScaleToFit() {
        std::cout << "Default Scale\n";
    }
};

class PictureMethod : public IMethod {
public:
    virtual void ScaleToFit() {
        std::cout << "Picture was scaled!\n";
    }
};

class IRenderableNode {
public:
    IRenderableNode(std::shared_ptr<IMethod> method) :
        method_(method) {}
    virtual void Transform() = 0;

    // Now ScaleToFit is not a part of interface to redefine it.
    // it will be instantiated at runtime
    void ScaleToFit(){
        method_->ScaleToFit();
    }
protected:
    std::shared_ptr<IMethod> method_;
};

class Picture : public IRenderableNode {
    using IRenderableNode::IRenderableNode;

    void Transform() {}

};