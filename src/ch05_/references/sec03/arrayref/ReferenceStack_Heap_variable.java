package ch05_.references.sec03.arrayref;
/*
 * 배열 : 동일한 자료형의 모음
 */
public class ReferenceStack_Heap_variable {
	public static void main(String[] args) {
		// 아래 배열 변수들은 모두 stack memory에 저장 
		//  new int[] 는 heap memory 에 { 1, 2, 3 } 배열 객체를 생성하고 이 배열 객체 참조 주소인 arr1에 저장
		// arr2 라는 또다른 공간에 배열객체를 생성하므로 arr1이 참조하는 배열객체와는 다름
		// arr3 = arr2 -> arr2가 참조하고 있는 배열 객체의 참조 주소를 arr3에 저장하므로 동일한 배열 객체를 참조 
		
		int[] arr1; //배열 변수 arr1 선언 [stack 영역(메서드 호출과 관련된 프레임을 저장하는 메모리 공간)
		// -> heap 동적개체와 배열을 공간에 저장(new-> 모든 객체를 만드는 행위)]
		int[] arr2; //배열 변수 arr2 선언
		int[] arr3; //배열 변수 arr3 선언
			
		arr1 = new int[] { 1, 2, 3 }; //배열 { 1, 2, 3 }을 생성하고 arr1 변수에 대입
		arr2 = new int[] { 1, 2, 3 }; //배열 { 1, 2, 3 }을 생성하고 arr2 변수에 대입
		arr3 = arr2; //배열 변수 arr2의 값을 배열 변수 arr3에 대입
			
		System.out.println(arr1 == arr2); // arr1과 arr2 변수가 같은 배열을 참조하는지 검사
		// 서로 다른 두 배열 객체를 참조 -> 즉 두 배열은 heap memory 에서 서로 다른 위치에 저장되어 있음 
		System.out.println(arr2 == arr3); // arr2와 arr3 변수가 같은 배열을 참조하는지 검사
		// 두 변수는 같은 메모리 주소를 가짐. 
	}
}
/*
 * Heap:
주소1: [1, 2, 3]  <- arr1가 참조
주소2: [1, 2, 3]  <- arr2와 arr3가 참조

Stack:
arr1 -> 주소1
arr2 -> 주소2
arr3 -> 주소2
*/
