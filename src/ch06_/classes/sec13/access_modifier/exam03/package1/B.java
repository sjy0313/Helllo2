78package ch06_.classes.sec13.access_modifier.exam03.package1;

public class B {
	public void method() {
		//객체 생성
		A a = new A();

		//필드값 변경
		a.field1 = 1; 		// o
		a.field2 = 1; 		// o default : 같은 패키지
		//a.field3 = 1;		// x private

		//메소드 호출
		a.method1(); 		// o public 
		a.method2(); 		// o default : 같은 패키지
		//a.method3(); 		// x
	}
}