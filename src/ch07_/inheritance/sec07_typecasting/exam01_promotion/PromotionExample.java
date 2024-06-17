package ch07_.inheritance.sec07_typecasting.exam01_promotion;
/* 자동 타입 변환(promotion) 부모타입 변수 = 자식타입객체;
 * 부모가 아니더라도 상승 계층에서 상위 타입이라면 promotion발생
 */
class A {
}

class B extends A {
}

class C extends A {
}

class D extends B {
}

class E extends C {
}

public class PromotionExample {
	public static void main(String[] args) {
		B b = new B();
		C c = new C();
		D d = new D();
		E e = new E();
		
		// 부모는 모든 자식 객체를 받을 수 있다 
		A a1 = b;
		A a2 = c;
		A a3 = d;
		A a4 = e;
		
		B b1 = d; // B는 D의 부모 : 가능
		C c1 = e; // C는 E의 부모 : 가능
		
		// 불가능 : e,d 상속관계에 있지 않다 
		// B b3 = e;
		// C c2 = d;
	}
}