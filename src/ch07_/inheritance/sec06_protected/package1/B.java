package ch07_.inheritance.sec06_protected.package1;
/*
 * A 클래스는 같은 패키지이기 떄문에 멤버들이 protected 여도 접근가능
 */
public class B {
	//메소드 선언
	public void method() {
		A a = new A();		//o
		a.field = "value"; 	//o
		a.method(); 			//o
	}
	public static void main(String[] args) {
		B b = new B();
		b.method();
		
		(new B()).method();
		
	}
	
}