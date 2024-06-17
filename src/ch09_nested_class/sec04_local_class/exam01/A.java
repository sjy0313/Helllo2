package ch09_nested_class.sec04_local_class.exam01;
// 클래스가 여러 관계를 맺는 경우 독립적으로 선언
// 특정 클래스와 관계를 맺는 경우 중첩 클래스(클래스 내부에 선언한 클래스) 선언하여 유지보수에 유용하게함. 

//로컬클래스인 경우 A$1B.class  / 

public class A {
	//생성자
	A() {
		//로컬 클래스 선언 (a 객체를 생성하여 b 객체 생성할 수 있음)
		class B { } // 같은 패키징에서만 B 클래스 사용할 수 있음.
		// private class B {} a 클래스 내부에서만 b 클래스를 사용할 수 있다

		//로컬 객체 생성
		B b = new B();
	}

	//메소드
	void method() {
		//로컬 클래스 선언
		class B { }

		//로컬 객체 생성
		B b = new B();
	}
}