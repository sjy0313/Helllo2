package ch07_.inheritance.sec07_typecasting.exam02_promotion;

public class ChildExample3 {
	public static void main(String[] args) {
		//자식 객체 생성해서 부모 타입으로 받음
		Parent parent = new Child();
		// 부모 타입으로 promotion 된 자식 객체는 부모 필드와 메소드로만
		// 접근이 가능.
		//비록 변수는 자식 객체를 참조하지만 변수로 접근 가능한 멤버는 
		// 부모 클래스 매버로 한정됨. 
		
		//메소드 호출
		parent.method1();
		parent.method2();
		//The method method3() is undefined for the type Parent
		//parent.method3(); //(호출 불가능)
		
		//강제 타입 변환 : 정상
		// 원래 객체는 Child 로 생성 되었기 때문에 
		// 자식타입 변수 = (자식타입)부모타입객체;
		
		Child child = (Child)parent;
		// 만약 자식 타입에 선언된 필드+메소드를 사용해야 한다면
		// 강제 타입 변환을 해서 다시 자식 타입으로 변환 필용
		child.method3();
	
		// 원래 객체는 Parent로 생성되었기 때문에 오류
		
		/*
		 * Parent-method1()
Child-method2()
Child-method3()
		 */
	}
}