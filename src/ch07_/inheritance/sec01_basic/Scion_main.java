package ch07_.inheritance.sec01_basic;

public class Scion_main {

	public static void main(String[] args) {
		Parent parent = new Parent();
		Child child = new Child();
		Scion scion = new Scion();
		System.out.println(parent == child); // false
		System.out.println(child); // ch07_.inheritance.sec01.Child@28a418fc
		System.out.println(parent); // ch07_.inheritance.sec01.Parent@5305068a
		// 값이 다름. 
		// false
		System.out.println(parent instanceof Parent); // true
		System.out.println(child instanceof Parent); // true
		System.out.println(scion instanceof Parent); // true
		System.out.println(scion instanceof Child); // true
		// child 객체는 부모로 부터의 연장선임을 알려줌.
		
		// java : instanceof / python : isinstnace
		// isinstance(object, classinfo) object : 검사할 객체 / classinfo(class들의 tuple)
		/* class MyClass:
	    pass

	    obj = MyClass()
		 print(isinstance(obj, MyClass))  # True
		print(isinstance(obj, object))   # True
		다중클래스의 검사(java에서는 불가능)
		print(isinstance(obj, (MyClass, int)))  # True, */
	}

}
